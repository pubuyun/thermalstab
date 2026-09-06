"""Render figures from explicit synthetic CSV fixtures, with no SPURS dependency."""

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thermal.common import load_config, write_csv, write_json, write_yaml
from thermal.pipeline import SINGLE_FIELDS
from thermal.search import BEAM_FIELDS
from spurs_plotting import main, read_beam_layers


def make_results(folder):
    """Synthetic scores only. Used for figure tests and visual inspection."""
    cfg = load_config(ROOT / "spurs_config.yaml")
    cfg["beam"]["depth"] = 10
    cfg["gradient"]["diversity"] += [3, 3]
    cfg["gradient"]["marginal"] += [-.2, -.2]
    write_yaml(folder / "config.resolved.yaml", cfg)
    rows = []
    for position in range(1, 5):
        for aa, score in (("A", 0), ("C", -.9 + position * .1), ("D", -.7 + position * .1)):
            rows.append(dict(zip(SINGLE_FIELDS, [f"A{position}{aa}", f"A{position}{aa}", "A", position, "", position,
                "A", aa, position < 4, score, "single", position < 4 and aa != "A" and score < -.5, "ok", ""])))
    write_csv(folder / "single_all.csv", SINGLE_FIELDS, rows)
    selected = sorted((row for row in rows if row["passes_single_cutoff"]), key=lambda row: row["spurs_score"])
    write_csv(folder / "single_candidates.csv", SINGLE_FIELDS, selected)
    pair_fields = ["mutation_i_original", "mutation_j_original", "mutation_i_model", "mutation_j_model", "spurs_score", "epistasis", "category", "status"]
    pairs = [
        ("A1C", "A2C", -2, -.5, "high_confidence", "ok"),
        ("A1C", "A2D", -.2, 1.5, "severe", "ok"),
        ("A1C", "A3C", -1, .7, "intermediate", "ok"),
        ("A2C", "A2D", None, None, "severe", "same_position_not_scored"),
        ("A2C", "A3C", -1.8, .2, "high_confidence", "ok"),
        ("A2D", "A3C", -1.1, .7, "intermediate", "ok"),
        ("A2D", "A4C", "nan", "nan", "intermediate", "error"),
    ]
    write_csv(folder / "pair_graph.csv", pair_fields, (dict(zip(pair_fields, [i, j, i, j, score, epi, category, status])) for i, j, score, epi, category, status in pairs))
    for depth in (2, 10):
        group = "/".join(f"A{i}C" for i in range(1, depth + 1))
        row = dict(zip(BEAM_FIELDS, [1, group, group, depth, -depth, "multi", 1, "high_confidence", "" if depth == 2 else "/".join(group.split("/")[:-1]), None if depth == 2 else -.4, "ok", ""]))
        write_csv(folder / f"beam_depth_{depth}.csv", BEAM_FIELDS, [row])
        write_json(folder / f"beam_depth_{depth}_summary.json", {"frequency_cap": 60})
    write_csv(folder / "beam_depth_3.csv", BEAM_FIELDS, [])


class PlotTests(unittest.TestCase):
    def test_all_figures_render_and_reports_distinguish_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_results(folder)
            before = {path.name: path.read_bytes() for path in folder.glob("*.csv")}
            with redirect_stdout(io.StringIO()):
                code = main([str(folder), "--sample-size", "3", "--matrix-size", "2", "--formats", "png", "pdf"])
            self.assertEqual(code, 0)
            report = json.loads((folder / "plots/plot_report.json").read_text())
            self.assertEqual(set(item["status"] for item in report["plots"].values()), {"completed"})
            pairs = report["plots"]["pairs"]
            self.assertEqual(pairs["class_counts"], {"severe": 2, "intermediate": 2, "high_confidence": 2})
            self.assertEqual((pairs["scored_pairs"], pairs["scatter_sample_size"], pairs["same_position_unscored"], pairs["skipped_rows"]), (5, 3, 1, 1))
            self.assertEqual(report["plots"]["beam"]["depths"], [2, 3, 10])
            self.assertEqual(report["plots"]["beam"]["selected_counts"], {"2": 1, "3": 0, "10": 1})
            self.assertEqual(len(list((folder / "plots").glob("*.png"))), 6)
            self.assertEqual(len(list((folder / "plots").glob("*.pdf"))), 6)
            for path in (folder / "plots").glob("*.png"):
                self.assertGreater(path.stat().st_size, 10000)
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            for path in (folder / "plots").glob("*.pdf"):
                self.assertTrue(path.read_bytes().startswith(b"%PDF"))
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.glob("*.csv")})

    def test_only_single_results_and_legacy_json_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_results(folder)
            for path in folder.glob("beam_depth_*"):
                path.unlink()
            (folder / "pair_graph.csv").unlink()
            (folder / "config.resolved.yaml").unlink()
            write_json(folder / "config.resolved.json", {"single": {"cutoff": -.5}})
            with redirect_stdout(io.StringIO()):
                code = main([str(folder), "--formats", "svg"])
            self.assertEqual(code, 0)
            report = json.loads((folder / "plots/plot_report.json").read_text())
            self.assertEqual(report["plots"]["pairs"]["status"], "skipped")
            self.assertEqual(report["plots"]["beam"]["status"], "skipped")
            self.assertEqual(len(list((folder / "plots").glob("*.svg"))), 2)

    def test_empty_beam_and_empty_pair_graph_render_without_fake_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_csv(folder / "beam_depth_12.csv", BEAM_FIELDS, [])
            write_csv(folder / "pair_graph.csv", ["mutation_i_model", "mutation_j_model", "spurs_score", "epistasis", "category", "status"], [])
            with redirect_stdout(io.StringIO()):
                code = main([str(folder), "--formats", "png"])
            self.assertEqual(code, 0)
            self.assertEqual(read_beam_layers(folder)[12]["rows"], [])
            report = json.loads((folder / "plots/pairs_plot_report.json").read_text())
            self.assertEqual(report["scored_pairs"], 0)


class CliTests(unittest.TestCase):
    def test_required_config_and_output_location_from_another_directory(self):
        script = ROOT / "scripts/run_spurs.py"
        missing = subprocess.run([sys.executable, str(script), "--validate-inputs"], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--config", missing.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / "small.experiment.yml"
            cfg = load_config(ROOT / "spurs_config.yaml")
            cfg["beam"]["depth"] = 4
            write_yaml(path, cfg)
            result = subprocess.run([sys.executable, str(script), "--config", str(path), "--validate-inputs"], cwd=folder.parent, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            output = path.with_suffix("")
            self.assertEqual(json.loads((output / "input_validation.json").read_text())["mutable_positions"], 453)
            self.assertFalse((output / "positive_controls.csv").exists())


if __name__ == "__main__":
    unittest.main()
