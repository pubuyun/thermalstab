"""YAML method configuration, provenance and atomic output helpers."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path


AA = "ACDEFGHIKLMNPQRSTVWY"
METHOD_SECTIONS = ("single", "pair_graph", "beam", "gradient")


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite prediction: {value}")
    return value


def chunks(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML support requires PyYAML: python -m pip install PyYAML") from exc
    return yaml


def read_yaml(path):
    yaml = yaml_module()

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=UniqueLoader)


def write_yaml(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml_module().safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def method_config(cfg):
    return {section: cfg[section] for section in METHOD_SECTIONS}


def exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")


def numeric(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def load_config(path):
    """Read only the public method parameters; execution settings are separate."""
    path = Path(path).resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--config must be a .yaml or .yml file")
    cfg = read_yaml(path)
    exact_keys(cfg, METHOD_SECTIONS, "Configuration")
    for section, keys in {
        "single": {"cutoff"},
        "beam": {"depth", "width", "high_confidence_quota", "high_edge_ratio", "frequency_limit"},
        "gradient": {"diversity", "marginal"},
        "pair_graph": {"severe", "high_confidence"},
    }.items():
        exact_keys(cfg[section], keys, section)
    beam = cfg["beam"]
    for key, lower in (("depth", 2), ("width", 1)):
        if type(beam[key]) is not int or beam[key] < lower:
            raise ValueError(f"beam.{key} must be an integer >= {lower}")
    for key in ("high_confidence_quota", "high_edge_ratio", "frequency_limit"):
        if not 0 < numeric(beam[key], f"beam.{key}") <= 1:
            raise ValueError(f"beam.{key} must be in (0, 1]")
    if math.floor(beam["frequency_limit"] * beam["width"]) < 1:
        raise ValueError("Frequency cap is zero; increase beam.width")
    for key in ("diversity", "marginal"):
        values = cfg["gradient"][key]
        if not isinstance(values, list) or len(values) < beam["depth"]:
            raise ValueError(f"gradient.{key} must cover beam.depth={beam['depth']} (index = depth - 1)")
        for depth, value in enumerate(values, 1):
            numeric(value, f"gradient.{key}[{depth - 1}]")
            if key == "diversity" and (type(value) is not int or not 0 <= value <= depth):
                raise ValueError("Diversity values must be integers between 0 and their depth")
            if key == "marginal" and value > 0:
                raise ValueError("Marginal thresholds must be nonpositive")
    for kind, pair_key in (("severe", "pair_ddg_min"), ("high_confidence", "pair_ddg_max")):
        rule = cfg["pair_graph"][kind]
        exact_keys(rule, {pair_key, "epistasis_max"}, f"pair_graph.{kind}")
        for key, value in rule.items():
            numeric(value, f"pair_graph.{kind}.{key}")
    numeric(cfg["single"]["cutoff"], "single.cutoff")
    return cfg


def resolve_run_config(config_path, *, pdb, mutable_positions, chain="A"):
    """Attach derived paths and environment values for internal use, not YAML."""
    cfg = load_config(config_path)
    output = Path(config_path).resolve().with_suffix("")
    if not output.name or output.is_file():
        raise ValueError(f"Result directory is not available: {output}")
    if not isinstance(chain, str) or len(chain) != 1:
        raise ValueError("Exactly one PDB chain is required")
    cfg["input"] = {"pdb": str(Path(pdb).resolve()), "mutable_positions": str(Path(mutable_positions).resolve()), "chain": chain}
    cfg["output_dir"] = str(output)
    runtime = {
        "spurs_repo": os.environ.get("SPURS_REPO", "/root/software/SPURS"),
        "offline_root": os.environ.get("SPURS_OFFLINE_ROOT", "/root/models/spurs-offline"),
        "model_revision": os.environ.get("SPURS_MODEL_REVISION", "0cc7a565af8f31eb122819f95a9d16e27b3d1596"),
        "device": os.environ.get("SPURS_DEVICE", "cuda"),
        "batch_size": int(os.environ.get("SPURS_BATCH_SIZE", "8")),
        "seed": int(os.environ.get("SPURS_SEED", "42")),
        "wt_zero_tolerance": 1e-5,
    }
    for key in ("spurs_repo", "offline_root"):
        runtime[key] = str(resolve_path(Path.cwd(), runtime[key]))
    if runtime["batch_size"] < 1 or not 0 <= runtime["seed"] < 2**32:
        raise ValueError("SPURS_BATCH_SIZE must be positive; SPURS_SEED must be in [0, 2**32)")
    if not runtime["model_revision"] or runtime["device"] != "cpu" and not runtime["device"].startswith("cuda"):
        raise ValueError("Specify SPURS_MODEL_REVISION and a cpu/cuda SPURS_DEVICE")
    cfg["runtime"] = runtime
    return cfg


def resolve_path(base, value):
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()
