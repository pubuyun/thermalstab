"""Adapter to the installed SPURS inference API, with explicit offline setup."""

import copy
import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import random
import subprocess
import sys

from .common import AA, digest_file, finite
from .structure import verify_model_mapping


LOG = logging.getLogger(__name__)


def verify_numpy_bridge(torch, numpy):
    """Fail before SPURS parsing when torch and the loaded NumPy cannot interoperate."""
    try:
        probe = numpy.zeros(1, dtype=numpy.float32)
        tensor = torch.from_numpy(probe)
        if tuple(tensor.shape) != (1,):
            raise RuntimeError(f"unexpected probe shape: {tuple(tensor.shape)}")
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            "PyTorch/NumPy bridge failed before SPURS inference: "
            f"torch={getattr(torch, '__version__', 'unknown')}, "
            f"numpy={getattr(numpy, '__version__', 'unknown')} from "
            f"{getattr(numpy, '__file__', 'unknown')}, python={sys.executable}. "
            "Use the same Python environment as the successful SPURS run, restore "
            "NumPy 1.26.4 with /root/software/SPURS/constraints-server.txt, then "
            "start a new Python process. Do not reinstall torch."
        ) from exc


def configure_environment(cfg):
    runtime = cfg["runtime"]
    root, repo = Path(runtime["offline_root"]), Path(runtime["spurs_repo"])
    if not (repo / "spurs/inference.py").is_file():
        raise FileNotFoundError(f"SPURS source missing: {repo}; set the SPURS_REPO environment variable")
    # These must be set before importing torch/huggingface_hub/spurs.
    os.environ["HF_HUB_CACHE"] = str(root / "hf_hub")
    os.environ["TORCH_HOME"] = str(root / "torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(repo))
    return root, repo


def preflight(cfg, residues):
    root, repo = configure_environment(cfg)
    runtime = cfg["runtime"]
    cache = root / "hf_hub/models--cyclization9--SPURS"
    revision = runtime["model_revision"]
    if (cache / "refs/main").read_text().strip() != revision:
        raise ValueError("Offline refs/main differs from the expected revision; check SPURS_MODEL_REVISION")
    files = []
    for model in ("spurs", "spurs_multi"):
        for rel in (".hydra/config.yaml", "checkpoints/best.ckpt"):
            files.append(cache / "snapshots" / revision / model / rel)
    files.extend(root / "torch/hub/checkpoints" / name for name in (
        "esm2_t33_650M_UR50D.pt", "esm2_t33_650M_UR50D-contact-regression.pt"
    ))
    fingerprints = {}
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty offline model file: {path}")
        stat = path.stat()
        fingerprints[str(path.relative_to(root))] = {
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest_file(path) if path.suffix == ".yaml" else None,
        }
    try:
        import torch
        import numpy
        import spurs.inference as inference
        from omegaconf import OmegaConf
        from spurs.datamodules.datasets.utils import alt_parse_PDB
    except ImportError as exc:
        raise RuntimeError("SPURS dependency import failed. Use the existing server Python and resolve the named missing dependency; see SPURS_AI_HANDOFF.md") from exc
    actual = Path(inference.__file__).resolve()
    if actual != (repo / "spurs/inference.py").resolve():
        raise RuntimeError(f"Unexpected installed SPURS source: {actual}")
    verify_numpy_bridge(torch, numpy)
    device = torch.device(runtime["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
        # Also exercises the installed torch build on this GPU architecture.
        probe = torch.ones((16, 16), device=device)
        if not bool(torch.isfinite(probe @ probe).all()):
            raise RuntimeError("GPU arithmetic preflight failed")
        gpu = torch.cuda.get_device_name(device)
    else:
        gpu = None
    for model in ("spurs", "spurs_multi"):
        model_cfg = OmegaConf.load(cache / "snapshots" / revision / model / ".hydra/config.yaml")
        if model_cfg["model"].get("name", "esm2_t33_650M_UR50D") != "esm2_t33_650M_UR50D":
            raise ValueError("Model uses an ESM variant not covered by the offline manifest")
        raw = alt_parse_PDB(cfg["input"]["pdb"], cfg["input"]["chain"])[0]
        batch = inference.parse_pdb(cfg["input"]["pdb"], "1CXI", cfg["input"]["chain"], model_cfg, device="cpu")
        verify_model_mapping(residues, raw, batch)
        if not bool(batch["mask"].bool().all()):
            raise ValueError("SPURS masked one or more backbone residues")
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    versions = {"python": sys.version, "torch": torch.__version__, "numpy": numpy.__version__, "cuda": torch.version.cuda}
    for package in ("spurs", "omegaconf", "huggingface-hub", "lmdb", "atom3d", "tqdm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    LOG.info("Offline cache, dependency imports, GPU and both model input mappings: PASS")
    return {
        "versions": versions, "gpu": gpu, "spurs_commit": git("rev-parse", "HEAD"),
        "spurs_branch": git("branch", "--show-current"), "spurs_git_status": git("status", "--porcelain"),
        "spurs_source_sha256": {str(p.relative_to(repo)): digest_file(p) for p in sorted((repo / "spurs").rglob("*.py"))},
        "model_revision": revision, "model_files": fingerprints,
        "model_identity_policy": "revision + sizes + mtimes; YAML SHA256; large weights are not content-hashed",
    }


class SpursBackend:
    def __init__(self, cfg, residues, kind):
        import torch
        from spurs import inference
        from spurs.datamodules.datasets.utils import alt_parse_PDB

        self.torch, self.inference = torch, inference
        self.device = cfg["runtime"]["device"]
        self.kind, self.residues = kind, residues
        self.batch_size = cfg["runtime"]["batch_size"]
        self.seed = cfg["runtime"]["seed"]
        self.wt_tolerance = cfg["runtime"]["wt_zero_tolerance"]
        LOG.info("Loading %s model once on %s", kind, self.device)
        loader = inference.get_SPURS_from_hub if kind == "single" else inference.get_SPURS_multi_from_hub
        self.model, model_cfg = loader(device=self.device)
        self.model.float().eval()
        self._seed()
        self.base = inference.parse_pdb(cfg["input"]["pdb"], "1CXI", cfg["input"]["chain"], model_cfg, device=self.device)
        raw = alt_parse_PDB(cfg["input"]["pdb"], cfg["input"]["chain"])[0]
        verify_model_mapping(residues, raw, self.base)
        if not bool(self.base["mask"].bool().all()):
            raise ValueError("Invalid SPURS backbone mask")
        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _seed(self):
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.torch.manual_seed(self.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.seed)

    def single_matrix(self):
        if self.kind != "single":
            raise ValueError("single_matrix requires the single model")
        self._seed()
        with self.torch.no_grad():
            # Upstream forward mutates its dictionary and some nested values.
            output = self.model(copy.deepcopy(self.base), return_logist=True).detach().cpu()
        if tuple(output.shape) != (len(self.residues), 20) or not bool(self.torch.isfinite(output).all()):
            raise ValueError(f"Invalid single prediction matrix: {tuple(output.shape)}")
        matrix = output.tolist()
        if any(abs(row[AA.index(residue.wt_aa)]) > self.wt_tolerance for row, residue in zip(matrix, self.residues)):
            raise ValueError("Wild-type columns are not zero; check model/API compatibility")
        self.log_memory()
        return matrix

    def score(self, groups):
        if self.kind != "multi":
            raise ValueError("Combination scores require the multi model")
        if not groups:
            return []
        if len({len(g) for g in groups}) != 1:
            raise ValueError("Each SPURS forward must use the same mutation count")
        sequence = self.base["seq"]
        for group in groups:
            positions = set()
            for mutation in group:
                position = int(mutation[1:-1])
                if not 1 <= position <= len(sequence) or mutation[0] != sequence[position - 1] or mutation[-1] not in AA or mutation[0] == mutation[-1] or position in positions:
                    raise ValueError(f"Invalid mutation group: {group}")
                positions.add(position)
        if len(groups) > self.batch_size:
            return sum((self.score(groups[i:i+self.batch_size]) for i in range(0, len(groups), self.batch_size)), [])
        try:
            return self._forward(groups)
        except self.torch.cuda.OutOfMemoryError:
            if len(groups) == 1:
                raise
            self.batch_size = max(1, len(groups) // 2)
            LOG.warning("GPU OOM: reducing combination batch size to %d", self.batch_size)
        # Leave the except block first so the traceback releases the failed
        # forward's tensor references before we attempt a smaller batch.
        gc.collect()
        self.torch.cuda.empty_cache()
        return self.score(groups)

    def _forward(self, groups):
        self._seed()
        with self.torch.no_grad():
            batch = copy.deepcopy(self.base)
            positions, amino_acids = self.inference.parse_pdb_for_mutation([list(g) for g in groups])
            batch["mut_ids"] = positions.to(self.device)
            batch["append_tensors"] = amino_acids.to(self.device)
            output = self.model(batch).detach().cpu().reshape(-1)
        if len(output) != len(groups):
            raise ValueError(f"Multi output count mismatch: {len(output)} != {len(groups)}")
        return [finite(value) for value in output.tolist()]

    def log_memory(self):
        if str(self.device).startswith("cuda"):
            LOG.info("Peak CUDA allocated memory: %.2f GiB", self.torch.cuda.max_memory_allocated(self.device) / 2**30)

    def close(self):
        self.log_memory()
        self.model = self.base = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
