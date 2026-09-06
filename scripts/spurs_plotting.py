"""Offline figures from completed CSV snapshots; never import SPURS or torch."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import re

from thermal.common import AA, read_yaml, write_json


CATEGORIES = ("severe", "intermediate", "high_confidence")
COLORS = {"severe": "#c44e52", "intermediate": "#d69c32", "high_confidence": "#27877d"}


def csv_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{Path(path).name}: missing columns {sorted(set(required) - set(reader.fieldnames or []))}")
        yield from reader


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def is_true(value):
    return str(value).strip().lower() in {"true", "1"}


def read_parameters(folder):
    # Older result folders remain plottable; no inference or migration is needed.
    if (folder / "config.resolved.yaml").is_file():
        return read_yaml(folder / "config.resolved.yaml")
    if (folder / "config.resolved.json").is_file():
        return json.loads((folder / "config.resolved.json").read_text(encoding="utf-8"))
    if (folder / "manifest.json").is_file():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("identity", {}).get("config", {})
    return {}


def save_figure(fig, destination, stem, formats):
    paths = []
    for extension in formats:
        path = destination / f"{stem}.{extension}"
        temporary = destination / f".{stem}.{extension}.tmp"
        fig.savefig(temporary, format=extension, dpi=180, bbox_inches="tight", facecolor="white")
        temporary.replace(path)
        paths.append(path.name)
    return paths


def empty_panel(ax, message):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#666666")


def plot_single(folder, destination, cfg, formats, plt, np):
    path = folder / "single_all.csv"
    if not path.is_file():
        return {"status": "skipped", "reason": "single_all.csv is not available"}
    required = {"model_position_1based", "pdb_resseq", "wt_aa", "mt_aa", "mutable", "spurs_score", "passes_single_cutoff", "status"}
    cells, positions, eligible, selected, best = {}, {}, [], [], {}
    skipped = 0
    for row in csv_rows(path, required):
        score = number(row["spurs_score"])
        position_value = number(row["model_position_1based"])
        if row["status"] != "ok" or score is None or position_value is None or not position_value.is_integer() or row["mt_aa"] not in AA:
            skipped += 1
            continue
        position = int(position_value)
        positions[position] = row["pdb_resseq"] + row.get("insertion_code", "")
        if not is_true(row["mutable"]) or row["wt_aa"] == row["mt_aa"]:
            continue
        key = (position, AA.index(row["mt_aa"]))
        if key in cells:
            raise ValueError(f"Duplicate single-mutation cell at model position {position}")
        cells[key] = score
        eligible.append(score)
        best[position] = min(best.get(position, score), score)
        if is_true(row["passes_single_cutoff"]):
            selected.append(score)
    cutoff = number(cfg.get("single", {}).get("cutoff"))
    if cutoff is None and (folder / "single_summary.json").is_file():
        cutoff = number(json.loads((folder / "single_summary.json").read_text())["cutoff"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    if eligible:
        bins = np.histogram_bin_edges(eligible, bins=min(60, max(8, int(math.sqrt(len(eligible))))))
        axes[0].hist(eligible, bins=bins, color="#91a8c0", label=f"All mutable substitutions (n={len(eligible):,})")
        axes[0].hist(selected, bins=bins, color=COLORS["high_confidence"], label=f"Selected (n={len(selected):,})")
        ordered = sorted(best)
        axes[1].scatter(ordered, [best[p] for p in ordered], c=[COLORS["high_confidence"] if cutoff is not None and best[p] < cutoff else "#91a8c0" for p in ordered], s=12)
        if cutoff is not None:
            axes[0].axvline(cutoff, color="#2e3641", ls="--", lw=1, label=f"Cutoff {cutoff:g}")
            axes[1].axhline(cutoff, color="#2e3641", ls="--", lw=1)
        axes[0].legend(fontsize=8)
    else:
        for ax in axes:
            empty_panel(ax, "No valid mutable substitutions")
    axes[0].set(xlabel="SPURS single score", ylabel="Substitutions", title="Single-mutation screening")
    axes[1].set(xlabel="Model sequence position (1-based)", ylabel="Best non-WT score at position", title="Position-level potential")
    files = save_figure(fig, destination, "single_scores", formats)
    plt.close(fig)
    if positions:
        ordered = sorted(positions)
        index = {position: i for i, position in enumerate(ordered)}
        matrix = np.full((20, len(ordered)), np.nan)
        for (position, aa), score in cells.items():
            matrix[aa, index[position]] = score
        bound = max((abs(v) for v in cells.values()), default=1) or 1
        cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#e9edf0")
        fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
        displayed = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-bound, vmax=bound, interpolation="nearest")
        ticks = np.unique(np.linspace(0, len(ordered) - 1, min(14, len(ordered))).astype(int))
        ax.set_xticks(ticks, [positions[ordered[i]] for i in ticks])
        ax.set_yticks(range(20), list(AA))
        ax.set(xlabel="PDB residue number (columns follow model order)", ylabel="Mutant amino acid", title="Single substitutions | gray = protected, WT or unavailable")
        fig.colorbar(displayed, ax=ax, label="SPURS single score")
        files += save_figure(fig, destination, "single_substitution_heatmap", formats)
        plt.close(fig)
    return {"status": "completed", "source": path.name, "eligible_substitutions": len(eligible), "selected": len(selected), "skipped_rows": skipped, "files": files}


def matrix_nodes(folder, limit):
    path = folder / "single_candidates.csv"
    if not path.is_file():
        return [], {}, "first encountered mutations"
    values, labels = {}, {}
    for row in csv_rows(path, {"mutation_model", "mutation_original", "spurs_score", "status"}):
        score = number(row["spurs_score"])
        if row["status"] == "ok" and score is not None:
            values[row["mutation_model"]] = score
            labels[row["mutation_model"]] = row["mutation_original"]
    nodes = sorted(values, key=lambda key: (values[key], key))[:limit]
    return nodes, labels, "best single-score mutations"


def plot_pairs(folder, destination, cfg, formats, plt, np, sample_size, matrix_size):
    path = folder / "pair_graph.csv"
    if not path.is_file():
        return {"status": "skipped", "reason": "pair_graph.csv is not available"}
    nodes, labels, selection = matrix_nodes(folder, matrix_size)
    fixed_nodes = bool(nodes)
    lookup = {name: i for i, name in enumerate(nodes)}
    matrix = np.full((matrix_size, matrix_size), np.nan)
    counts, sample, total_scored, unscored, skipped = Counter(), [], 0, 0, 0
    rng = random.Random(42)
    required = {"mutation_i_model", "mutation_j_model", "spurs_score", "epistasis", "category", "status"}
    for row in csv_rows(path, required):
        category = row["category"]
        if category not in CATEGORIES:
            skipped += 1
            continue
        if row["status"] == "same_position_not_scored":
            if category != "severe":
                raise ValueError("Same-position pair must be severe")
            unscored += 1
        elif row["status"] == "ok":
            score, epistasis = number(row["spurs_score"]), number(row["epistasis"])
            if score is None or epistasis is None:
                skipped += 1
                continue
            total_scored += 1
            point = (score, epistasis, category)
            if len(sample) < sample_size:
                sample.append(point)
            else:
                index = rng.randrange(total_scored)
                if index < sample_size:
                    sample[index] = point
        else:
            skipped += 1
            continue
        counts[category] += 1
        for side in ("i", "j"):
            name = row[f"mutation_{side}_model"]
            if name in lookup:
                labels.setdefault(name, row.get(f"mutation_{side}_original", name))
            if not fixed_nodes and name not in lookup and len(nodes) < matrix_size:
                lookup[name] = len(nodes)
                nodes.append(name)
                labels[name] = row.get(f"mutation_{side}_original", name)
        i, j = lookup.get(row["mutation_i_model"]), lookup.get(row["mutation_j_model"])
        if i is not None and j is not None:
            matrix[i, j] = matrix[j, i] = CATEGORIES.index(category)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    bars = axes[0].bar(["Severe", "Intermediate", "High confidence"], [counts[c] for c in CATEGORIES], color=[COLORS[c] for c in CATEGORIES])
    axes[0].bar_label(bars, labels=[f"{counts[c]:,}" for c in CATEGORIES], padding=3)
    axes[0].margins(y=.15)
    axes[0].set(ylabel="Pairs (all valid rows)", title=f"Compatibility classes | same-position pairs: {unscored:,}")
    for category in CATEGORIES:
        points = [(score, epi) for score, epi, label in sample if label == category]
        if points:
            x, y = zip(*points)
            axes[1].scatter(x, y, s=8, alpha=.4, color=COLORS[category], label=category.replace("_", " "), rasterized=True)
    if sample:
        axes[1].legend(fontsize=8, markerscale=2)
    else:
        empty_panel(axes[1], "No scored pairs")
    for category, pair_key, color in (("severe", "pair_ddg_min", COLORS["severe"]), ("high_confidence", "pair_ddg_max", COLORS["high_confidence"])):
        rule = cfg.get("pair_graph", {}).get(category, {})
        if number(rule.get(pair_key)) is not None:
            axes[1].axvline(rule[pair_key], color=color, ls="--", lw=1)
        if number(rule.get("epistasis_max")) is not None:
            axes[1].axhline(rule["epistasis_max"], color=color, ls="--", lw=1)
    axes[1].set(xlabel="SPURS multi score (pair)", ylabel="Pair score - single i - single j", title=f"Score vs. epistasis | uniform sample {len(sample):,}/{total_scored:,}")
    files = save_figure(fig, destination, "pair_scores_and_classes", formats)
    plt.close(fig)
    if nodes:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap([COLORS[c] for c in CATEGORIES]).with_extremes(bad="#e9edf0")
        fig, ax = plt.subplots(figsize=(max(7, len(nodes) * .24), max(6, len(nodes) * .22)), layout="constrained")
        shown = ax.imshow(np.ma.masked_invalid(matrix[:len(nodes), :len(nodes)]), cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), interpolation="nearest")
        text = [labels.get(node, node) for node in nodes]
        ax.set_xticks(range(len(nodes)), text, rotation=90, fontsize=7)
        ax.set_yticks(range(len(nodes)), text, fontsize=7)
        ax.set_title(f"Pair compatibility | {len(nodes)} {selection}\nGray = diagonal or unavailable; subset only", fontsize=10)
        colorbar = fig.colorbar(shown, ax=ax, ticks=[0, 1, 2], fraction=.04)
        colorbar.ax.set_yticklabels(["Severe", "Intermediate", "High confidence"])
        files += save_figure(fig, destination, "pair_compatibility_matrix", formats)
        plt.close(fig)
    return {"status": "completed", "source": path.name, "class_counts": dict(counts), "scored_pairs": total_scored,
            "same_position_unscored": unscored, "scatter_sample_size": len(sample), "sample_seed": 42,
            "matrix_nodes_model": nodes, "skipped_rows": skipped, "files": files}


def read_beam_layers(folder):
    layers = {}
    for path in folder.glob("beam_depth_*.csv"):
        match = re.fullmatch(r"beam_depth_(\d+)\.csv", path.name)
        if not match:
            continue
        depth, rows, skipped = int(match[1]), [], 0
        for row in csv_rows(path, {"mutation_model", "spurs_score", "high_edge_ratio", "pool", "marginal", "status"}):
            score, ratio = number(row["spurs_score"]), number(row["high_edge_ratio"])
            members = row["mutation_model"].split("/")
            if row["status"] != "ok" or score is None or ratio is None or not 0 <= ratio <= 1 or len(members) != depth or len(set(members)) != depth:
                skipped += 1
                continue
            rows.append({**row, "score_value": score, "ratio_value": ratio, "marginal_value": number(row["marginal"]), "members": members})
        summary_path = folder / f"beam_depth_{depth}_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        layers[depth] = {"rows": rows, "skipped": skipped, "summary": summary}
    return dict(sorted(layers.items()))


def plot_beam(folder, destination, cfg, formats, plt, np, top_mutations):
    layers = read_beam_layers(folder)
    if not layers:
        return {"status": "skipped", "reason": "No beam_depth_*.csv files are available"}
    depths = list(layers)
    active = [depth for depth in depths if layers[depth]["rows"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    score_ax, count_ax, ratio_ax, marginal_ax = axes.flat
    if active:
        score_ax.boxplot([[row["score_value"] for row in layers[d]["rows"]] for d in active], positions=active, widths=.5, showfliers=False)
        score_ax.plot(active, [min(row["score_value"] for row in layers[d]["rows"]) for d in active], color=COLORS["high_confidence"], marker="o", ms=4, label="Best in layer")
        ratio_ax.boxplot([[row["ratio_value"] for row in layers[d]["rows"]] for d in active], positions=active, widths=.5, showfliers=False)
        score_ax.legend(fontsize=8)
    else:
        empty_panel(score_ax, "No selected combinations")
        empty_panel(ratio_ax, "No selected combinations")
    high = [sum(row["pool"] == "high_confidence" for row in layers[d]["rows"]) for d in depths]
    middle = [sum(row["pool"] == "intermediate" for row in layers[d]["rows"]) for d in depths]
    count_ax.bar(depths, high, color=COLORS["high_confidence"], label="High confidence")
    count_ax.bar(depths, middle, bottom=high, color=COLORS["intermediate"], label="Intermediate")
    width = number(cfg.get("beam", {}).get("width"))
    if width is not None:
        count_ax.axhline(width, color="#2e3641", ls="--", lw=1, label=f"Width {width:g}")
    count_ax.legend(fontsize=8)
    q_cutoff = number(cfg.get("beam", {}).get("high_edge_ratio"))
    if q_cutoff is not None:
        ratio_ax.axhline(q_cutoff, color=COLORS["high_confidence"], ls="--", lw=1)
    marginal_depths = [d for d in depths if any(row["marginal_value"] is not None for row in layers[d]["rows"])]
    if marginal_depths:
        marginal_ax.boxplot([[row["marginal_value"] for row in layers[d]["rows"] if row["marginal_value"] is not None] for d in marginal_depths], positions=marginal_depths, widths=.5, showfliers=False)
        thresholds = cfg.get("gradient", {}).get("marginal", [])
        known = [d for d in marginal_depths if d <= len(thresholds) and number(thresholds[d - 1]) is not None]
        if known:
            marginal_ax.plot(known, [thresholds[d - 1] for d in known], color=COLORS["severe"], ls="--", marker=".", label="Marginal threshold")
            marginal_ax.legend(fontsize=8)
    else:
        empty_panel(marginal_ax, "No scored extensions yet")
    for ax in axes.flat:
        ax.set_xticks(depths)
        ax.set_xlim(min(depths) - .6, max(depths) + .6)
        ax.set_xlabel("Mutation count / depth")
    score_ax.set(title="Selected total-score distributions", ylabel="SPURS multi score")
    count_ax.set(title="Beam occupancy", ylabel="Selected combinations")
    ratio_ax.set(title="High-confidence edge ratio", ylabel="q(S)", ylim=(-.03, 1.05))
    marginal_ax.set(title="Added-mutation contribution", ylabel="Child score - selected parent score")
    files = save_figure(fig, destination, "beam_progress", formats)
    plt.close(fig)

    counts = {d: Counter(m for row in layers[d]["rows"] for m in row["members"]) for d in depths}
    selected_counts = {d: len(layers[d]["rows"]) for d in depths}
    all_mutations = set().union(*(set(counter) for counter in counts.values()))
    # Rank across layers by relative prevalence, so wider layers do not dominate.
    ranked = sorted(all_mutations, key=lambda m: (-max(counts[d][m] / max(1, selected_counts[d]) for d in depths), m))[:top_mutations]
    if ranked:
        prevalence = np.array([[counts[d][mutation] / selected_counts[d] if selected_counts[d] else np.nan for d in depths] for mutation in ranked])
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(ranked) * .23)), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
        cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#e9edf0")
        shown = axes[0].imshow(np.ma.masked_invalid(prevalence), aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_xticks(range(len(depths)), depths)
        axes[0].set_yticks(range(len(ranked)), ranked, fontsize=8)
        axes[0].set(xlabel="Depth", ylabel="Mutation (model numbering)", title=f"Top {len(ranked)} mutation prevalence\nFraction of actual selected combinations")
        fig.colorbar(shown, ax=axes[0], label="Selected fraction", fraction=.04)
        maximum = [max(counts[d].values(), default=0) for d in depths]
        axes[1].plot(depths, maximum, marker="o", color="#406b9c", label="Observed maximum count")
        cap = [number(layers[d]["summary"].get("frequency_cap")) for d in depths]
        known = [i for i, value in enumerate(cap) if value is not None]
        if known:
            axes[1].plot([depths[i] for i in known], [cap[i] for i in known], ls="--", color=COLORS["severe"], label="Allowed count (configured B)")
        elif width is not None and number(cfg.get("beam", {}).get("frequency_limit")) is not None:
            axes[1].axhline(math.floor(width * cfg["beam"]["frequency_limit"]), ls="--", color=COLORS["severe"], label="Allowed count (configured B)")
        axes[1].set_xticks(depths)
        axes[1].set(xlabel="Depth", ylabel="Combinations containing one mutation", title="Mutation frequency constraint")
        axes[1].legend(fontsize=8)
        files += save_figure(fig, destination, "beam_mutation_frequency", formats)
        plt.close(fig)
    return {"status": "completed", "depths": depths, "selected_counts": selected_counts,
            "skipped_rows": sum(layer["skipped"] for layer in layers.values()), "files": files}


def main(argv=None, default_plot="all"):
    parser = argparse.ArgumentParser(description="Plot SPURS result snapshots without a GPU or SPURS installation")
    parser.add_argument("results", type=Path, help="Result folder, including old JSON-config runs")
    parser.add_argument("--plots", nargs="+", choices=("all", "single", "pairs", "beam"), default=[default_plot])
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument("--sample-size", type=int, default=20000, help="Maximum uniform pair-scatter sample; category counts still use every row")
    parser.add_argument("--matrix-size", type=int, default=40, help="Maximum mutations in the compatibility matrix")
    parser.add_argument("--top-mutations", type=int, default=30, help="Maximum rows in mutation-frequency heatmap")
    args = parser.parse_args(argv)
    if args.sample_size < 1 or not 1 <= args.matrix_size <= 100 or not 1 <= args.top_mutations <= 100:
        parser.error("sample-size must be positive; matrix-size and top-mutations must be in [1, 100]")
    folder = args.results.resolve()
    if not folder.is_dir():
        parser.error(f"Result folder does not exist: {folder}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        parser.exit(1, f"Plotting dependency missing: {exc}. Install requirements-tools.txt\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    destination = folder / "plots"
    destination.mkdir(exist_ok=True)
    try:
        cfg = read_parameters(folder)
    except Exception as exc:
        parser.exit(1, f"Cannot read saved run parameters: {exc}\n")
    kinds = ["single", "pairs", "beam"] if "all" in args.plots else list(dict.fromkeys(args.plots))
    report = {"results": str(folder), "plots": {}, "note": "Only completed CSV snapshots; no model inference. Missing stages are skipped."}
    for kind in kinds:
        try:
            shared = (folder, destination, cfg, list(dict.fromkeys(args.formats)), plt, np)
            if kind == "single":
                result = plot_single(*shared)
            elif kind == "pairs":
                result = plot_pairs(*shared, args.sample_size, args.matrix_size)
            else:
                result = plot_beam(*shared, args.top_mutations)
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            plt.close("all")
        report["plots"][kind] = result
        print(f"{kind}: {result['status']} | {result.get('reason', result.get('error', ', '.join(result.get('files', []))))}")
    # Keep per-stage reports when scripts are invoked separately.
    for kind, result in report["plots"].items():
        write_json(destination / f"{kind}_plot_report.json", result)
    write_json(destination / "plot_report.json", report)
    statuses = [item["status"] for item in report["plots"].values()]
    return 1 if "error" in statuses or "completed" not in statuses else 0
