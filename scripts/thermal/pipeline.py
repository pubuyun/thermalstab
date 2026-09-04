"""CLI orchestration for stages 2 (singles) and 3 (combinations)."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

from .backend import SpursBackend, preflight
from .common import AA, digest_file, digest_object, finite, load_config, write_csv, write_json
from .search import beam_search, build_pair_graph
from .storage import CachedScorer, Store, output_lock
from .structure import Mutation, parse_original_group, read_mutable, read_structure, write_mapping


LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SINGLE_FIELDS = ["mutation_original", "mutation_model", "chain", "pdb_resseq", "insertion_code",
                 "model_position_1based", "wt_aa", "mt_aa", "mutable", "spurs_score", "model_kind",
                 "passes_single_cutoff", "status", "error"]
GROUP_FIELDS = ["mutation_original", "mutation_model", "mutation_count", "spurs_score", "model_kind", "status", "error"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def bind_run(cfg, environment, output):
    inputs = {key: digest_file(cfg["input"][key]) for key in ("pdb", "mutable_positions")}
    source = {p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    source["run_spurs.py"] = digest_file(ROOT / "scripts/run_spurs.py")
    identity = {
        "schema": 1, "config": cfg, "input_sha256": inputs, "pipeline_source_sha256": source,
        "spurs_source_sha256": environment["spurs_source_sha256"],
        "model_files": environment["model_files"], "versions": environment["versions"],
        "gpu": environment["gpu"], "spurs_commit": environment["spurs_commit"],
    }
    fingerprint = digest_object(identity)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Run identity changed (config/input/code/model/environment). Use a new output_dir; old predictions are not reusable under this identity")
    else:
        if (output / "predictions.sqlite3").exists() or (output / "single_matrix.json").exists():
            raise ValueError("Unbound prediction files exist without manifest.json; use a new output_dir")
        write_json(manifest_path, {"created_at": utc_now(), "fingerprint": fingerprint, "identity": identity, "environment": environment})
    write_json(output / "config.resolved.json", cfg)


def single_stage(cfg, residues, mutable, output):
    matrix_path = output / "single_matrix.json"
    if matrix_path.exists():
        LOG.info("Reusing completed single scan")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))["scores"]
    else:
        backend = SpursBackend(cfg, residues, "single")
        try:
            matrix = backend.single_matrix()
        finally:
            backend.close()
        write_json(matrix_path, {"alphabet": AA, "scores": matrix, "model_kind": "single"})
    if len(matrix) != len(residues) or any(len(row) != 20 for row in matrix):
        raise ValueError("Cached single matrix has an invalid shape")
    for row, residue in zip(matrix, residues):
        for value in row:
            finite(value)
        if abs(row[AA.index(residue.wt_aa)]) > cfg["runtime"]["wt_zero_tolerance"]:
            raise ValueError("Cached single matrix WT column is not zero")
    mutable_set = set(mutable)
    scan_rows, selected = [], []
    for residue, row in zip(residues, matrix):
        for aa, score in zip(AA, row):
            mutation = Mutation(residue, aa)
            passes = residue in mutable_set and aa != residue.wt_aa and score < cfg["single"]["cutoff"]
            scan_rows.append({
                "mutation_original": mutation.original, "mutation_model": mutation.model,
                "chain": residue.chain, "pdb_resseq": residue.pdb_resseq, "insertion_code": residue.insertion_code,
                "model_position_1based": residue.model_position_1based, "wt_aa": residue.wt_aa, "mt_aa": aa,
                "mutable": residue in mutable_set, "spurs_score": score, "model_kind": "single",
                "passes_single_cutoff": passes, "status": "ok", "error": "",
            })
            if passes:
                selected.append((mutation, score))
    scan_rows.sort(key=lambda row: (row["spurs_score"], row["model_position_1based"], row["mt_aa"]))
    write_csv(output / "single_all.csv", SINGLE_FIELDS, scan_rows)
    write_csv(output / "single_candidates.csv", SINGLE_FIELDS, (row for row in scan_rows if row["passes_single_cutoff"]))
    selected.sort(key=lambda pair: (pair[1], pair[0].residue.model_position_1based, pair[0].mt_aa))
    write_json(output / "single_summary.json", {
        "residues": len(residues), "mutable_positions": len(mutable), "evaluated_mutable_substitutions": 19 * len(mutable),
        "selected_mutations": len(selected), "cutoff": cfg["single"]["cutoff"], "comparison": "strictly_less_than",
    })
    LOG.info("Single scan: %d mutable substitutions, %d selected (< %s)", 19 * len(mutable), len(selected), cfg["single"]["cutoff"])
    return [m for m, _ in selected], [score for _, score in selected]


def score_groups(groups, residues, scorer, output_path):
    # PDB WT validation is deliberately independent of the mutable list: controls
    # may include a protected position, but are never injected into the search.
    parsed = [parse_original_group(group, residues) for group in groups]
    scores = {}
    for count in sorted({len(group) for group in parsed}):
        indices = [i for i, group in enumerate(parsed) if len(group) == count]
        values = scorer.score([[m.model for m in parsed[i]] for i in indices])
        scores.update(zip(indices, values))
    rows = [{
        "mutation_original": "/".join(m.original for m in group),
        "mutation_model": "/".join(m.model for m in group), "mutation_count": len(group),
        "spurs_score": scores[i], "model_kind": "multi", "status": "ok", "error": "",
    } for i, group in enumerate(parsed)]
    write_csv(output_path, GROUP_FIELDS, rows)
    return rows


def smoke_test(cfg, residues, controls, output):
    single = SpursBackend(cfg, residues, "single")
    try:
        single.single_matrix()
    finally:
        single.close()
    LOG.info("Single prediction: PASS")
    multi = SpursBackend(cfg, residues, "multi")
    try:
        control = parse_original_group(controls[0], residues) if controls else tuple(
            Mutation(r, "A" if r.wt_aa != "A" else "C") for r in residues[:2]
        )
        group = [m.model for m in control]
        alternate = [m.model for m in control[:-1]] + [Mutation(control[-1].residue, next(
            aa for aa in AA if aa not in {control[-1].residue.wt_aa, control[-1].mt_aa}
        )).model]
        first = multi.score([group])[0]
        batch = multi.score([group, alternate])
        again = multi.score([group])[0]
        if abs(first - batch[0]) > 1e-4 or abs(first - again) > 1e-4:
            raise ValueError("Multi predictions changed with batch/repeated forward; check upstream mutation/state handling")
    finally:
        multi.close()
    LOG.info("Multi prediction: PASS")
    write_json(output / "smoke_test.json", {"status": "passed", "time": utc_now(), "first": first, "batch": batch, "again": again})
    LOG.info("SPURS installation and target-PDB inference verified.")


def run(args, cfg, residues, mutable, output):
    for group in cfg["positive_controls"]:
        parse_original_group(group, residues)
    if args.validate_inputs:
        write_mapping(output / "input_mapping_unverified.csv", residues)
        write_json(output / "input_validation.json", {
            "status": "input_only_passed", "model_mapping_verified": False,
            "residues": len(residues), "mutable_positions": len(mutable),
            "positive_controls": cfg["positive_controls"],
        })
        LOG.info("Input-only validation: PASS (%d residues, %d mutable); SPURS model mapping not yet checked", len(residues), len(mutable))
        return
    environment = preflight(cfg, residues)
    if args.check:
        write_json(output / "preflight.json", {"status": "passed", "time": utc_now(), "environment": environment})
        write_mapping(output / "preflight_mapping.csv", residues)
        LOG.info("Preflight passed. Neural-network forward is checked by --smoke-test.")
        return
    bind_run(cfg, environment, output)
    write_mapping(output / "residue_mapping.csv", residues)
    if args.smoke_test:
        smoke_test(cfg, residues, cfg["positive_controls"], output)
        return
    store = Store(output / "predictions.sqlite3")
    try:
        mutations, singles = [], []
        if args.stage != "score":
            mutations, singles = single_stage(cfg, residues, mutable, output)
        if args.stage == "single":
            return
        backend = SpursBackend(cfg, residues, "multi")
        try:
            scorer = CachedScorer(store, backend, cfg["runtime"]["batch_size"])
            score_groups(cfg["positive_controls"], residues, scorer, output / "positive_controls.csv")
            if args.stage == "score":
                lines = Path(args.mutations).read_text(encoding="utf-8-sig").splitlines()
                groups = [line.split("#", 1)[0].strip().split("/") for line in lines if line.split("#", 1)[0].strip()]
                if not groups or any(len(group) < 2 or len(group) > 8 for group in groups):
                    raise ValueError("Mutation file must contain groups of 2–8 mutations, separated by '/', one group per line")
                score_groups(groups, residues, scorer, output / "specified_combinations.csv")
                return
            graph = build_pair_graph(mutations, singles, cfg, store, scorer, output)
            if args.stage != "pairs":
                beam_search(graph, cfg, store, scorer, output)
        finally:
            backend.close()
    finally:
        try:
            store.export_failures(output / "failed_predictions.csv")
        finally:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="SPURS single scan, full pair graph and diverse beam search; no other evaluation methods.")
    parser.add_argument("--config", type=Path, default=ROOT / "spurs_config.json")
    parser.add_argument("--stage", choices=("all", "single", "pairs", "search", "score"), default="all")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--validate-inputs", action="store_true", help="Only standard-library PDB/config/WT checks; no model or CUDA needed")
    checks.add_argument("--check", action="store_true", help="Offline cache, imports, GPU and real SPURS parser checks")
    checks.add_argument("--smoke-test", action="store_true", help="Real single and multi GPU inference, repeated-forward and batching checks")
    parser.add_argument("--mutations", type=Path, help="PDB-numbered combinations for --stage score")
    args = parser.parse_args(argv)
    if args.stage == "score" and not args.mutations:
        parser.error("--stage score requires --mutations")
    if args.mutations and args.stage != "score":
        parser.error("--mutations is only used with --stage score")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        cfg = load_config(args.config)
        output = Path(cfg["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        with output_lock(output):
            handler = logging.FileHandler(output / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(handler)
            try:
                write_json(output / "run_status.json", {"status": "running", "started_at": utc_now(), "arguments": vars(args) | {"config": str(args.config), "mutations": str(args.mutations) if args.mutations else None}})
                residues = read_structure(cfg["input"]["pdb"], cfg["input"]["chain"])
                mutable = read_mutable(cfg["input"]["mutable_positions"], residues)
                run(args, cfg, residues, mutable, output)
                write_json(output / "run_status.json", {"status": "completed", "finished_at": utc_now(), "stage": args.stage,
                    "mode": "validate_inputs" if args.validate_inputs else "check" if args.check else "smoke_test" if args.smoke_test else "prediction"})
            except BaseException as exc:
                write_json(output / "run_status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "time": utc_now(), "error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
    except KeyboardInterrupt:
        LOG.error("Interrupted. Completed prediction batches are saved; rerun the same command to resume.")
        return 130
    except Exception:
        LOG.exception("Pipeline failed; no missing/error score is treated as a stabilizing mutation")
        return 1
    return 0
