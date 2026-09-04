"""Explicit PDB-to-model mapping; reject ambiguous or incomplete structures."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re

from .common import AA, write_csv


THREE_TO_ONE = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(), AA
))


@dataclass(frozen=True)
class Residue:
    chain: str
    pdb_resseq: int
    insertion_code: str
    wt_aa: str
    model_position_1based: int

    @property
    def pdb_label(self):
        return f"{self.pdb_resseq}{self.insertion_code}"


@dataclass(frozen=True)
class Mutation:
    residue: Residue
    mt_aa: str

    @property
    def original(self):
        return f"{self.residue.wt_aa}{self.residue.pdb_label}{self.mt_aa}"

    @property
    def model(self):
        return f"{self.residue.wt_aa}{self.residue.model_position_1based}{self.mt_aa}"


def read_structure(path, chain):
    atoms, names, models = {}, {}, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("MODEL"):
            models += 1
            if models > 1:
                raise ValueError("Multiple PDB models are unsupported; supply a single structure")
        if line[:6] not in {"ATOM  ", "HETATM"} or line[21:22] != chain:
            continue
        if line.startswith("HETATM"):
            if line[17:20] == "MSE":
                raise ValueError("MSE requires an explicitly prepared standard-residue PDB")
            continue
        key = (int(line[22:26]), line[26:27].strip())
        if key[1]:
            raise ValueError(f"Insertion code at {chain}:{key}; current SPURS parser cannot safely map it")
        name, atom = line[17:20].strip(), line[12:16].strip()
        if name not in THREE_TO_ONE:
            raise ValueError(f"Nonstandard residue {name} at {key}")
        if key in names and names[key] != name:
            raise ValueError(f"Conflicting residue identities at {key}")
        names[key] = name
        if atom not in {"N", "CA", "C", "O"}:
            continue
        if line[16:17].strip():
            raise ValueError(f"Ambiguous backbone alternate location at {key}; prepare PDB explicitly")
        xyz = tuple(float(line[i:i+8]) for i in (30, 38, 46))
        if not all(math.isfinite(x) for x in xyz) or float(line[54:60]) <= 0:
            raise ValueError(f"Invalid backbone coordinates/occupancy at {key}")
        if atom in atoms.setdefault(key, {}):
            raise ValueError(f"Duplicate backbone atom at {key}: {atom}")
        atoms[key][atom] = xyz
    keys = sorted(names)
    if not keys:
        raise ValueError(f"No protein residues in chain {chain}")
    if [k[0] for k in keys] != list(range(keys[0][0], keys[-1][0] + 1)):
        raise ValueError("Internal residue numbering gaps: SPURS inserts '-' and cannot scan all 20 amino acids safely")
    residues = []
    for position, key in enumerate(keys, 1):
        if set(atoms.get(key, {})) != {"N", "CA", "C", "O"}:
            raise ValueError(f"Missing backbone atoms at {chain}:{key}")
        residues.append(Residue(chain, key[0], key[1], THREE_TO_ONE[names[key]], position))
    return residues


def read_mutable(path, residues):
    lookup = {r.pdb_label: r for r in residues}
    selected = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        label = line.split("#", 1)[0].strip()
        if not label:
            continue
        if label not in lookup or label in selected:
            raise ValueError(f"Unknown/duplicate mutable PDB position at line {line_number}: {label}")
        selected.add(label)
    if not selected:
        raise ValueError("No mutable positions")
    return [r for r in residues if r.pdb_label in selected]


def parse_original_group(group, residues):
    lookup = {r.pdb_label: r for r in residues}
    parsed, positions = [], set()
    for text in group:
        match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])(-?\d+)([ACDEFGHIKLMNPQRSTVWY])", text)
        if not match:
            raise ValueError(f"Invalid PDB mutation: {text}")
        wt, position, mt = match.groups()
        residue = lookup.get(str(int(position)))
        if residue is None or residue.wt_aa != wt or wt == mt or residue.model_position_1based in positions:
            raise ValueError(f"Invalid WT/position or conflicting mutation: {text}")
        positions.add(residue.model_position_1based)
        parsed.append(Mutation(residue, mt))
    return tuple(sorted(parsed, key=lambda m: m.residue.model_position_1based))


def verify_model_mapping(residues, raw, batch):
    expected = "".join(r.wt_aa for r in residues)
    if raw["seq"] != expected or batch["seq"] != expected:
        raise ValueError("PDB sequence does not exactly match SPURS raw parser and batch['seq']")
    if list(map(str, raw["resn_list"])) != [r.pdb_label for r in residues]:
        raise ValueError("SPURS resn_list does not match explicit PDB mapping")


def write_mapping(path, residues):
    write_csv(path, ["chain", "pdb_resseq", "insertion_code", "wt_aa", "model_position_1based"], (asdict(r) for r in residues))
