"""Complete pair graph and deterministic quota-constrained diverse beam search."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import logging
import math
from pathlib import Path
import time

from .common import chunks, finite, write_csv, write_json
from .storage import group_key


LOG = logging.getLogger(__name__)
UNKNOWN, SEVERE, INTERMEDIATE, HIGH = 0, 1, 2, 3
EDGE_NAMES = {SEVERE: "severe", INTERMEDIATE: "intermediate", HIGH: "high_confidence"}


def classify_pair(score, single_i, single_j, same_position, cfg):
    if same_position:
        return SEVERE, None
    score, single_i, single_j = map(finite, (score, single_i, single_j))
    epistasis = score - single_i - single_j
    severe, high = cfg["severe"], cfg["high_confidence"]
    if score > severe["pair_ddg_min"] or epistasis > severe["epistasis_max"]:
        return SEVERE, epistasis
    if score < high["pair_ddg_max"] and epistasis < high["epistasis_max"]:
        return HIGH, epistasis
    return INTERMEDIATE, epistasis


class PairGraph:
    def __init__(self, mutations):
        self.mutations = mutations
        self.index = {m.model: i for i, m in enumerate(mutations)}
        # One byte per pair (triangular), rather than millions of Python objects.
        self.edges = [bytearray(i) for i in range(len(mutations))]

    def set(self, i, j, category):
        if i == j:
            raise ValueError("A node cannot form a pair with itself")
        i, j = max(i, j), min(i, j)
        self.edges[i][j] = category

    def get(self, i, j):
        if i == j:
            return SEVERE
        category = self.edges[max(i, j)][min(i, j)]
        if category == UNKNOWN:
            raise ValueError(f"Incomplete compatibility graph at {i},{j}")
        return category

    def indices(self, key):
        return tuple(self.index[m] for m in key.split(";"))

    def high_ratio(self, indices):
        return sum(self.get(i, j) == HIGH for i, j in combinations(indices, 2)) / math.comb(len(indices), 2)

    def compatible_extension(self, indices, new):
        return all(self.get(old, new) != SEVERE for old in indices)


def build_pair_graph(mutations, singles, cfg, store, scorer, output):
    graph = PairGraph(mutations)
    store.reset_pool(2)
    counts = Counter()
    total = math.comb(len(mutations), 2)
    LOG.info("Complete pair graph: %d nodes, %d pairs; no top-K truncation", len(mutations), total)
    fields = ["mutation_i_original", "mutation_j_original", "mutation_i_model", "mutation_j_model",
              "spurs_score", "epistasis", "category", "status"]
    def rows():
        processed, last_log = 0, time.monotonic()
        for batch in chunks(combinations(range(len(mutations)), 2), max(128, scorer.batch_size)):
            to_score = [(i, j) for i, j in batch if mutations[i].residue != mutations[j].residue]
            scores = iter(scorer.score([[mutations[i].model, mutations[j].model] for i, j in to_score]))
            pool = []
            for i, j in batch:
                same = mutations[i].residue == mutations[j].residue
                score = None if same else next(scores)
                category, epistasis = classify_pair(score, singles[i], singles[j], same, cfg["pair_graph"])
                graph.set(i, j, category)
                counts[EDGE_NAMES[category]] += 1
                if category != SEVERE:
                    pool.append((2, group_key([mutations[i].model, mutations[j].model]), score, float(category == HIGH), None, None))
                yield {
                    "mutation_i_original": mutations[i].original, "mutation_j_original": mutations[j].original,
                    "mutation_i_model": mutations[i].model, "mutation_j_model": mutations[j].model,
                    "spurs_score": score, "epistasis": epistasis, "category": EDGE_NAMES[category],
                    "status": "same_position_not_scored" if same else "ok",
                }
            store.add_pool(pool)
            processed += len(batch)
            if time.monotonic() - last_log >= 30 or processed == total:
                LOG.info("Pair graph: %d/%d; new multi predictions=%d, cache hits=%d", processed, total, scorer.new_count, scorer.hit_count)
                last_log = time.monotonic()
    write_csv(Path(output) / "pair_graph.csv", fields, rows())
    write_json(Path(output) / "pair_graph_summary.json", {"nodes": len(mutations), "pairs": total, "categories": dict(counts)})
    return graph


@dataclass(frozen=True)
class BeamItem:
    key: str
    score: float
    ratio: float
    parent: str | None
    marginal: float | None

    @property
    def members(self):
        return frozenset(self.key.split(";"))


def select_beam(store, depth, cfg):
    beam_cfg = cfg["beam"]
    width = beam_cfg["width"]
    high_target = math.floor(width * beam_cfg["high_confidence_quota"])
    frequency_cap = math.floor(width * beam_cfg["frequency_limit"])
    min_distance = cfg["gradient"]["diversity"][depth - 1]
    selected, sets, frequencies = [], [], Counter()
    def take(rows, limit):
        for row in rows:
            if len(selected) >= limit:
                break
            item = BeamItem(*row)
            members = item.members
            if any(frequencies[m] >= frequency_cap for m in members):
                continue
            if any(depth - len(members & old) < min_distance for old in sets):
                continue
            selected.append(item)
            sets.append(members)
            frequencies.update(members)
    take(store.ordered_pool(depth, beam_cfg["high_edge_ratio"], True), high_target)
    high_selected = len(selected)
    # Only the documented fallback: intermediate candidates fill a high-pool shortfall.
    # No loosening of distance, frequency, score or compatibility constraints.
    take(store.ordered_pool(depth, beam_cfg["high_edge_ratio"], False), width)
    selected.sort(key=lambda item: (item.score, item.key))
    stats = {
        "depth": depth, "eligible_pool": store.pool_count(depth), "selected": len(selected),
        "high_confidence": high_selected, "intermediate": len(selected) - high_selected,
        "high_target": high_target, "frequency_cap": frequency_cap, "frequency_denominator": width,
        "min_distance": min_distance, "max_observed_frequency": max(frequencies.values(), default=0),
        "underfilled": len(selected) < width,
    }
    return selected, stats


def expand_layer(parents, depth, graph, cfg, store, scorer):
    store.reset_extensions()
    store.reset_pool(depth)
    pending = []
    for parent in parents:
        indices = graph.indices(parent.key)
        for new, mutation in enumerate(graph.mutations):
            if graph.compatible_extension(indices, new):
                key = group_key([*parent.key.split(";"), mutation.model])
                pending.append((key, parent.key, parent.score))
                if len(pending) >= 1000:
                    store.add_extensions(pending)
                    pending.clear()
    store.add_extensions(pending)
    total = store.db.execute("SELECT count(*) FROM extensions").fetchone()[0]
    LOG.info("Depth %d: %d unique compatible extensions from %d beam parents", depth, total, len(parents))
    processed, last_log = 0, time.monotonic()
    for batch in chunks(store.extensions(), max(128, scorer.batch_size)):
        scores = scorer.score([row[0].split(";") for row in batch])
        pool = []
        for (key, parent, parent_score), score in zip(batch, scores):
            marginal = score - parent_score
            if marginal <= cfg["gradient"]["marginal"][depth - 1]:
                pool.append((depth, key, score, graph.high_ratio(graph.indices(key)), parent, marginal))
        store.add_pool(pool)
        processed += len(batch)
        if time.monotonic() - last_log >= 30 or processed == total:
            LOG.info("Depth %d: scored %d/%d extensions", depth, processed, total)
            last_log = time.monotonic()


BEAM_FIELDS = ["rank", "mutation_original", "mutation_model", "mutation_count", "spurs_score",
               "model_kind", "high_edge_ratio", "pool", "parent_model", "marginal", "status", "error"]


def beam_rows(items, graph, high_ratio):
    for rank, item in enumerate(items, 1):
        indices = graph.indices(item.key)
        yield {
            "rank": rank, "mutation_original": "/".join(graph.mutations[i].original for i in indices),
            "mutation_model": item.key.replace(";", "/"), "mutation_count": len(indices),
            "spurs_score": item.score, "model_kind": "multi", "high_edge_ratio": item.ratio,
            "pool": "high_confidence" if item.ratio >= high_ratio else "intermediate",
            "parent_model": item.parent.replace(";", "/") if item.parent else "", "marginal": item.marginal,
            "status": "ok", "error": "",
        }


def beam_search(graph, cfg, store, scorer, output):
    output = Path(output)
    layers, previous = [], []
    all_rows = []
    for depth in range(2, cfg["beam"]["depth"] + 1):
        if depth > 2:
            expand_layer(previous, depth, graph, cfg, store, scorer)
        previous, stats = select_beam(store, depth, cfg)
        rows = list(beam_rows(previous, graph, cfg["beam"]["high_edge_ratio"]))
        write_csv(output / f"beam_depth_{depth}.csv", BEAM_FIELDS, rows)
        write_json(output / f"beam_depth_{depth}_summary.json", stats)
        layers.append(stats)
        all_rows.extend(rows)
        LOG.info("Depth %d selected %d: high=%d, intermediate=%d", depth, stats["selected"], stats["high_confidence"], stats["intermediate"])
        if not previous:
            LOG.info("Search exhausted at depth %d; constraints remain unchanged", depth)
            break
    # Clear stale derived files beyond an early stop (e.g. an interrupted old run).
    for depth in range(layers[-1]["depth"] + 1, cfg["beam"]["depth"] + 1):
        write_csv(output / f"beam_depth_{depth}.csv", BEAM_FIELDS, [])
        write_json(output / f"beam_depth_{depth}_summary.json", {"depth": depth, "status": "not_reached"})
    write_csv(output / "beam_all_depths.csv", BEAM_FIELDS, all_rows)
    write_csv(output / "final_candidates.csv", BEAM_FIELDS, list(beam_rows(previous, graph, cfg["beam"]["high_edge_ratio"])))
    write_json(output / "search_summary.json", {
        "status": "completed" if previous and layers[-1]["depth"] == cfg["beam"]["depth"] else "exhausted",
        "target_depth": cfg["beam"]["depth"], "layers": layers,
        "last_nonempty_depth": max((s["depth"] for s in layers if s["selected"]), default=None),
        "final_candidates_policy": "only candidates at target depth; empty when search exhausts",
    })
    return layers
