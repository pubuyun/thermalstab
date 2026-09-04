"""Standard-library configuration, provenance and atomic output helpers."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path


AA = "ACDEFGHIKLMNPQRSTVWY"


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


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"input", "output_dir", "runtime", "single", "pair_graph", "beam", "gradient", "positive_controls"}
    if set(cfg) != required:
        raise ValueError(f"Configuration sections must be exactly {sorted(required)}")
    for section, keys in {
        "input": {"pdb", "chain", "mutable_positions"},
        "runtime": {"spurs_repo", "offline_root", "model_revision", "device", "batch_size", "seed", "wt_zero_tolerance"},
        "single": {"cutoff"},
        "beam": {"depth", "width", "high_confidence_quota", "high_edge_ratio", "frequency_limit", "frequency_denominator"},
        "gradient": {"source", "diversity", "marginal"},
    }.items():
        if set(cfg[section]) != keys:
            raise ValueError(f"Unexpected/missing keys in {section}: expected {sorted(keys)}")
    if len(cfg["input"]["chain"]) != 1:
        raise ValueError("Exactly one PDB chain is required")
    for key in ("pdb", "mutable_positions"):
        cfg["input"][key] = str(resolve_path(path.parent, cfg["input"][key]))
    cfg["output_dir"] = str(resolve_path(path.parent, cfg["output_dir"]))
    for key in ("spurs_repo", "offline_root"):
        cfg["runtime"][key] = str(resolve_path(path.parent, cfg["runtime"][key]))
    beam, runtime = cfg["beam"], cfg["runtime"]
    for name, value, lower, upper in (
        ("depth", beam["depth"], 2, 8), ("width", beam["width"], 1, 100000),
        ("batch_size", runtime["batch_size"], 1, 100000), ("seed", runtime["seed"], 0, 2**32 - 1),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid {name}: {value}")
    for key in ("high_confidence_quota", "high_edge_ratio", "frequency_limit"):
        if not 0 < finite(beam[key]) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if beam["frequency_denominator"] != "configured_width":
        raise ValueError("frequency_denominator must be configured_width; constraints are never relaxed")
    if math.floor(beam["frequency_limit"] * beam["width"]) < 1:
        raise ValueError("Frequency cap is zero; increase beam.width")
    for key in ("diversity", "marginal"):
        values = cfg["gradient"][key]
        if len(values) != 8:
            raise ValueError(f"gradient.{key} must contain exactly 8 values, indexed by depth - 1")
        for depth, value in enumerate(values, 1):
            finite(value)
            if key == "diversity" and (type(value) is not int or not 0 <= value <= depth):
                raise ValueError("Invalid diversity threshold")
            if key == "marginal" and value > 0:
                raise ValueError("Marginal thresholds must be nonpositive")
    if set(cfg["pair_graph"]) != {"severe", "high_confidence"}:
        raise ValueError("Invalid pair_graph keys")
    for kind, pair_key in (("severe", "pair_ddg_min"), ("high_confidence", "pair_ddg_max")):
        rule = cfg["pair_graph"][kind]
        if set(rule) != {pair_key, "epistasis_max"}:
            raise ValueError(f"Invalid pair_graph.{kind} keys")
        for value in rule.values():
            finite(value)
    finite(cfg["single"]["cutoff"])
    if finite(runtime["wt_zero_tolerance"]) < 0:
        raise ValueError("wt_zero_tolerance must be nonnegative")
    if not isinstance(cfg["positive_controls"], list) or not all(isinstance(g, list) and len(g) >= 2 for g in cfg["positive_controls"]):
        raise ValueError("positive_controls must be a list of multi-mutation lists")
    if not runtime["model_revision"] or runtime["device"] != "cpu" and not runtime["device"].startswith("cuda"):
        raise ValueError("Specify model revision and cpu/cuda device")
    return cfg


def resolve_path(base, value):
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()
