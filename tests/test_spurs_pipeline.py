"""CPU-only behavioral tests. Synthetic scores are never production predictions."""

import argparse
from contextlib import nullcontext
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thermal.common import AA, load_config
from thermal.backend import SpursBackend
from thermal.pipeline import bind_run, run, single_stage
from thermal.search import (
    HIGH, INTERMEDIATE, SEVERE, BeamItem, PairGraph, beam_search,
    build_pair_graph, classify_pair, expand_layer, select_beam,
)
from thermal.storage import CachedScorer, Store
from thermal.structure import Mutation, Residue, parse_original_group, read_mutable, read_structure, verify_model_mapping


def default_config():
    return load_config(ROOT / "spurs_config.json")


def mutations(count):
    return [Mutation(Residue("A", i, "", "A", i), "C") for i in range(1, count + 1)]


class SyntheticBackend:
    def __init__(self, score=None):
        self.calls = []
        self.function = score or (lambda group: -len(group))

    def score(self, groups):
        self.calls.extend(tuple(g) for g in groups)
        return [self.function(group) for group in groups]


class StructureTests(unittest.TestCase):
    def test_real_inputs_and_control(self):
        residues = read_structure(ROOT / "1CXI.pdb", "A")
        mutable = read_mutable(ROOT / "mutable_positions.txt", residues)
        self.assertEqual((len(residues), len(mutable)), (686, 453))
        group = parse_original_group(["N188D", "K192R"], residues)
        self.assertEqual([m.model for m in group], ["N188D", "K192R"])
        self.assertNotIn(residues[191], mutable)  # Control bypasses mutable filtering only for evaluation.
        with self.assertRaises(ValueError):
            parse_original_group(["A188D", "K192R"], residues)
        with self.assertRaises(ValueError):
            parse_original_group(["N188D", "N188A"], residues)

    def test_parser_mapping_must_match_sequence_and_numbering(self):
        residues = [Residue("A", 21, "", "A", 1), Residue("A", 22, "", "C", 2)]
        verify_model_mapping(residues, {"seq": "AC", "resn_list": [21, 22]}, {"seq": "AC"})
        self.assertEqual(parse_original_group(["A21D", "C22E"], residues)[0].model, "A1D")
        with self.assertRaises(ValueError):
            verify_model_mapping(residues, {"seq": "AC", "resn_list": [1, 2]}, {"seq": "AC"})
        with self.assertRaises(ValueError):
            verify_model_mapping(residues, {"seq": "AC", "resn_list": [21, 22]}, {"seq": "CA"})

    def test_reject_missing_backbone_gap_and_insertion(self):
        original = (ROOT / "1CXI.pdb").read_text().splitlines()
        cases = [
            [line for line in original if not (line.startswith("ATOM") and line[21] == "A" and line[22:26].strip() == "2")],
            [line for line in original if not (line.startswith("ATOM") and line[21] == "A" and line[22:26].strip() == "2" and line[12:16].strip() == "CA")],
            [line[:26] + "B" + line[27:] if line.startswith("ATOM") and line[21] == "A" and line[22:26].strip() == "2" else line for line in original],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.pdb"
            for lines in cases:
                path.write_text("\n".join(lines))
                with self.assertRaises(ValueError):
                    read_structure(path, "A")


class RuleTests(unittest.TestCase):
    def test_document_arrays(self):
        cfg = default_config()
        self.assertEqual(cfg["gradient"]["diversity"], [0, 1, 1, 1, 2, 2, 2, 3])
        self.assertEqual(cfg["gradient"]["marginal"], [0, 0, -.5, -.5, -.3, -.3, -.2, -.2])

    def test_pair_strict_boundaries(self):
        cfg = default_config()["pair_graph"]
        def category(score, a=-.75, b=-.75):
            return classify_pair(score, a, b, False, cfg)[0]
        self.assertEqual(category(-.5), INTERMEDIATE)  # epistasis == 1 is not severe.
        self.assertEqual(category(-.499), SEVERE)
        self.assertEqual(category(-1.2), INTERMEDIATE)
        self.assertEqual(category(-1.20001), HIGH)
        self.assertEqual(category(-1.5, -1, -1), INTERMEDIATE)  # epistasis == .5
        self.assertEqual(category(-1.50001, -1, -1), HIGH)
        self.assertEqual(category(-2, -2, -2), SEVERE)  # good pair score, bad epistasis
        self.assertEqual(classify_pair(None, -1, -1, True, cfg), (SEVERE, None))
        with self.assertRaises(ValueError):
            category(float("nan"))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.store = Store(self.output / "cache.sqlite3")
        self.cfg = default_config()

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_cache_order_dedup_and_restart(self):
        backend = SyntheticBackend(lambda group: sum(int(m[1:-1]) for m in group))
        scorer = CachedScorer(self.store, backend, 2)
        self.assertEqual(scorer.score([["A3C", "A1C"], ["A1C", "A3C"], ["A2C", "A4C"]]), [4, 4, 6])
        self.assertEqual(len(backend.calls), 2)
        self.store.close()
        self.store = Store(self.output / "cache.sqlite3")
        resumed = CachedScorer(self.store, backend, 2)
        self.assertEqual(resumed.score([["A3C", "A1C"]]), [4])
        self.assertEqual(len(backend.calls), 2)

    def test_nonfinite_is_error_and_retried(self):
        scorer = CachedScorer(self.store, SyntheticBackend(lambda g: math.nan), 8)
        with self.assertRaises(ValueError):
            scorer.score([["A1C", "A2C"]])
        self.assertEqual(self.store.db.execute("SELECT score,status FROM predictions").fetchone(), (None, "error"))
        scorer.backend = SyntheticBackend()
        self.assertEqual(scorer.score([["A1C", "A2C"]]), [-2])
        self.assertEqual(self.store.db.execute("SELECT status FROM predictions").fetchone()[0], "ok")

    def test_all_pairs_and_same_site_conflict(self):
        nodes = mutations(4)
        nodes.append(Mutation(nodes[0].residue, "D"))
        backend = SyntheticBackend()
        scorer = CachedScorer(self.store, backend, 8)
        graph = build_pair_graph(nodes, [-.75] * 5, self.cfg, self.store, scorer, self.output)
        self.assertEqual(len(backend.calls), 9)
        with (self.output / "pair_graph.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        self.assertEqual(graph.get(0, 4), SEVERE)
        self.assertEqual(graph.get(1, 4), HIGH)
        self.assertFalse(graph.compatible_extension((0, 1), 4))

    def test_quota_fallback_and_global_ascending_order(self):
        self.cfg["beam"].update(width=10, frequency_limit=1)
        high = [(2, f"A{i}C;A{i+100}C", -i, 1, None, None) for i in range(1, 4)]
        middle = [(2, f"A{i}C;A{i+100}C", -i, 0, None, None) for i in range(10, 22)]
        self.store.add_pool(high + middle)
        chosen, stats = select_beam(self.store, 2, self.cfg)
        self.assertEqual((stats["high_confidence"], stats["intermediate"]), (3, 7))
        self.assertEqual([c.score for c in chosen], sorted(c.score for c in chosen))

    def test_distance_and_frequency_never_relaxed(self):
        self.cfg["beam"].update(width=10, high_confidence_quota=1)
        self.store.add_pool([(2, f"A1C;A{i}C", -i, 1, None, None) for i in range(2, 30)])
        chosen, stats = select_beam(self.store, 2, self.cfg)
        self.assertEqual(len(chosen), 6)  # 60% of configured width, not actual size.
        self.assertTrue(stats["underfilled"])
        self.store.add_pool([
            (5, "A1C;A2C;A3C;A4C;A5C", -10, 1, None, None),
            (5, "A1C;A2C;A3C;A4C;A6C", -9, 1, None, None),
            (5, "A1C;A2C;A3C;A7C;A8C", -8, 1, None, None),
        ])
        chosen, _ = select_beam(self.store, 5, self.cfg)
        self.assertEqual(len(chosen), 2)
        self.assertEqual([c.score for c in chosen], [-10, -8])

    def test_all_parent_paths_and_marginal_equality(self):
        nodes = mutations(3)
        graph = PairGraph(nodes)
        for i, j in combinations(range(3), 2):
            graph.set(i, j, HIGH)
        parents = [BeamItem("A1C;A2C", -3, 1, None, None), BeamItem("A1C;A3C", -2, 1, None, None)]
        backend = SyntheticBackend(lambda g: -2.5)
        scorer = CachedScorer(self.store, backend, 8)
        expand_layer(parents, 3, graph, self.cfg, self.store, scorer)
        row = self.store.db.execute("SELECT parent,marginal FROM pool WHERE depth=3").fetchone()
        self.assertEqual(row, ("A1C;A3C", -.5))
        self.assertEqual(len(backend.calls), 1)

    def test_nonadditive_pair_predictions_are_used(self):
        nodes = mutations(3)
        scorer = CachedScorer(self.store, SyntheticBackend(lambda g: 1), 8)
        graph = build_pair_graph(nodes, [-4] * 3, self.cfg, self.store, scorer, self.output)
        self.assertTrue(all(graph.get(i, j) == SEVERE for i, j in combinations(range(3), 2)))
        layers = beam_search(graph, self.cfg, self.store, scorer, self.output)
        self.assertEqual(layers[0]["selected"], 0)
        self.assertEqual(json.loads((self.output / "search_summary.json").read_text())["status"], "exhausted")

    def test_full_search_to_eight_and_resume(self):
        nodes = mutations(16)
        backend = SyntheticBackend()
        scorer = CachedScorer(self.store, backend, 8)
        graph = build_pair_graph(nodes, [-.75] * 16, self.cfg, self.store, scorer, self.output)
        layers = beam_search(graph, self.cfg, self.store, scorer, self.output)
        self.assertEqual(layers[-1]["depth"], 8)
        self.assertGreater(layers[-1]["selected"], 0)
        for layer in layers:
            self.assertLessEqual(layer["max_observed_frequency"], 60)
            with (self.output / f"beam_depth_{layer['depth']}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            sets = [set(row["mutation_model"].split("/")) for row in rows]
            self.assertTrue(all(layer["depth"] - len(a & b) >= layer["min_distance"] for a, b in combinations(sets, 2)))
        old_calls = len(backend.calls)
        old_final = (self.output / "final_candidates.csv").read_bytes()
        graph = build_pair_graph(nodes, [-.75] * 16, self.cfg, self.store, scorer, self.output)
        beam_search(graph, self.cfg, self.store, scorer, self.output)
        self.assertEqual(len(backend.calls), old_calls)
        self.assertEqual((self.output / "final_candidates.csv").read_bytes(), old_final)


class PipelineTests(unittest.TestCase):
    def test_all_stage_orchestration_with_synthetic_backend(self):
        cfg = default_config()
        residues = read_structure(ROOT / "1CXI.pdb", "A")
        mutable = read_mutable(ROOT / "mutable_positions.txt", residues)
        matrix = [[0 if aa == r.wt_aa else 1 for aa in AA] for r in residues]
        for residue in mutable[:16]:
            matrix[residue.model_position_1based - 1][next(i for i, aa in enumerate(AA) if aa != residue.wt_aa)] = -.75
        backends = []
        def factory(cfg, residues, kind):
            backend = SyntheticBackend()
            backend.single_matrix = lambda: matrix
            backend.close = lambda: None
            backends.append((kind, backend))
            return backend
        environment = {"spurs_source_sha256": {}, "model_files": {}, "versions": {}, "gpu": "synthetic", "spurs_commit": "synthetic"}
        with tempfile.TemporaryDirectory() as tmp, patch("thermal.pipeline.SpursBackend", side_effect=factory), patch("thermal.pipeline.preflight", return_value=environment):
            output = Path(tmp)
            cfg["output_dir"] = str(output)
            args = argparse.Namespace(validate_inputs=False, check=False, smoke_test=False, stage="all")
            run(args, cfg, residues, mutable, output)
            self.assertEqual([kind for kind, _ in backends], ["single", "multi"])
            self.assertEqual(json.loads((output / "search_summary.json").read_text())["status"], "completed")
            with (output / "positive_controls.csv").open() as handle:
                controls = list(csv.DictReader(handle))
            self.assertEqual(controls[0]["mutation_original"], "N188D/K192R")
            self.assertEqual(controls[0]["spurs_score"], "-2.0")
            with (output / "single_candidates.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 16)

    def test_single_cutoff_full_scan_and_control_not_injected(self):
        residues = [Residue("A", i, "", "A", i) for i in range(1, 4)]
        matrix = [[0] * 20 for _ in residues]
        matrix[0][AA.index("C")] = -.5
        matrix[0][AA.index("D")] = -.50001
        matrix[1][AA.index("D")] = -10  # protected position
        matrix[2][AA.index("C")] = -.6
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "single_matrix.json").write_text(json.dumps({"scores": matrix}))
            selected, scores = single_stage(default_config(), residues, [residues[0], residues[2]], output)
            self.assertEqual([m.model for m in selected], ["A3C", "A1D"])
            with (output / "single_all.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 60)

    def test_manifest_prevents_mixed_runs(self):
        cfg = default_config()
        environment = {"spurs_source_sha256": {}, "model_files": {}, "versions": {}, "gpu": "fake", "spurs_commit": "fake"}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            bind_run(cfg, environment, output)
            bind_run(cfg, environment, output)
            cfg["single"]["cutoff"] = -.7
            with self.assertRaises(ValueError):
                bind_run(cfg, environment, output)


class BackendContractTests(unittest.TestCase):
    def backend(self):
        backend = object.__new__(SpursBackend)
        backend.kind, backend.device, backend.batch_size = "multi", "cpu", 8
        backend.base = {"seq": "AAAA", "mut_ids": [0], "append_tensors": [0], "nested": {"value": [1]}}
        backend._seed = lambda: None
        backend.torch = SimpleNamespace(no_grad=nullcontext)
        return backend

    def test_forward_rebuilds_tensors_and_does_not_mutate_base(self):
        class Value:
            def __init__(self, values):
                self.values = values
            def to(self, device):
                return self
            def detach(self):
                return self
            def cpu(self):
                return self
            def reshape(self, shape):
                return self
            def tolist(self):
                return self.values
            def __len__(self):
                return len(self.values)
        backend = self.backend()
        backend.inference = SimpleNamespace(parse_pdb_for_mutation=lambda groups: (Value(groups), Value(groups)))
        def model(batch):
            size = len(batch["mut_ids"])
            batch["nested"]["value"].append(99)
            batch["append_tensors"] = "upstream changed tensor layout"
            batch["mut_ids"] = "upstream changed indices"
            return Value([-2] * size)
        backend.model = model
        self.assertEqual(backend.score([["A1C", "A2C"]]), [-2])
        self.assertEqual(backend.score([["A1C", "A2C"], ["A3C", "A4C"]]), [-2, -2])
        self.assertEqual(backend.base["nested"]["value"], [1])
        self.assertEqual(backend.base["append_tensors"], [0])

    def test_rejects_mixed_sizes_and_wrong_wt_before_forward(self):
        backend = self.backend()
        for groups in ([["C1D", "A2C"]], [["A1C", "A1D"]], [["A1C"], ["A2C", "A3C"]]):
            with self.assertRaises(ValueError):
                backend.score(groups)

    def test_oom_retries_after_leaving_exception_scope(self):
        class OutOfMemory(RuntimeError):
            pass
        backend = self.backend()
        calls, cleanup_exception = [], []
        backend.torch.cuda = SimpleNamespace(OutOfMemoryError=OutOfMemory, empty_cache=lambda: cleanup_exception.append(sys.exception()))
        def forward(groups):
            calls.append(len(groups))
            if len(groups) > 1:
                raise OutOfMemory("synthetic OOM")
            return [-2]
        backend._forward = forward
        self.assertEqual(backend.score([["A1C", "A2C"], ["A3C", "A4C"]]), [-2, -2])
        self.assertEqual(calls, [2, 1, 1])
        self.assertEqual(cleanup_exception, [None])


if __name__ == "__main__":
    unittest.main()
