"""Filter 1CXI residues using ConSurf and heavy-atom structural exclusions.

Writes only results/mutable_positions.txt (one PDB position per line).
Positions 183 and 370 are always excluded, in addition to structural and ConSurf filters.
Standard library only. Run from the project root:
    python scripts/filter_mutable_residues.py
"""

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path


AA = dict(zip(
    'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
    'ARNDCQEGHILKMFPSTWYV',
))


def read_consurf(path, chain):
    rows = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not re.match(r'^\s*\d+\s', line):
            continue
        fields = [value.strip() for value in line.split('\t')]
        if len(fields) != 10:
            raise ValueError(f'Unexpected ConSurf row: {line}')
        name, number, mapped_chain = fields[2].split(':')
        if mapped_chain != chain:
            continue
        match = re.fullmatch(r'(-?\d+)([A-Za-z]?)', number)
        if not match:
            raise ValueError(f'Unmapped PDB residue: {fields[2]}')
        key = (chain, int(match[1]), match[2])
        grade = int(fields[4].replace('*', '').strip())
        if key in rows or not 1 <= grade <= 9 or AA.get(name) != fields[1]:
            raise ValueError(f'Invalid/duplicate ConSurf mapping: {fields[2]}')
        rows[key] = dict(sequence_position=int(fields[0]), wild_type=fields[1],
                         residue_name=name, consurf_score=float(fields[3]),
                         consurf_grade=grade, insufficient_data='*' in fields[4],
                         confidence_interval=fields[5], exposure=fields[6],
                         functional_structural=fields[7], msa_data=fields[8],
                         residue_variety=fields[9])
    if not rows:
        raise ValueError('No ConSurf residues found')
    return rows


def read_pdb(path, chain, ligand_names):
    protein = defaultdict(list)
    ligands, calcium, bonds = [], [], []
    model_count = 0
    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == 'MODEL':
            model_count += 1
            if model_count > 1:
                raise ValueError('Multiple models: supply a single-model PDB')
        if record == 'SSBOND':
            bonds.append(((line[15], int(line[17:21]), line[21].strip()),
                          (line[29], int(line[31:35]), line[35].strip())))
        if record not in {'ATOM', 'HETATM'}:
            continue
        element = line[76:78].strip().upper()
        if not element:
            raise ValueError('PDB element field is required')
        if element in {'H', 'D', 'T'} or float(line[54:60]) <= 0:
            continue
        key = (line[21], int(line[22:26]), line[26].strip())
        name = line[17:20].strip()
        atom = dict(key=key, name=name, atom=line[12:16].strip(), altloc=line[16].strip(),
                    xyz=tuple(float(line[i:i+8]) for i in (30, 38, 46)))
        if record == 'ATOM' and key[0] == chain:
            protein[key].append(atom)
        elif record == 'HETATM' and element == 'CA':
            calcium.append(atom)
        elif record == 'HETATM' and name in ligand_names:
            ligands.append(atom)
        elif record == 'HETATM' and name not in {'HOH', 'DOD', 'WAT'}:
            raise ValueError(f'Unclassified heterogen {name}; specify --ligands explicitly')
    if not protein or not ligands or not calcium:
        raise ValueError('Protein, ligand and calcium atoms must all be present')
    return protein, ligands, calcium, bonds


def atom_label(atom):
    chain, number, insertion = atom['key']
    return f"{chain}:{atom['name']}{number}{insertion}:{atom['atom']}" + (
        f":alt{atom['altloc']}" if atom['altloc'] else '')


def nearest(atoms, targets):
    distance, source, target = min(
        (math.dist(a['xyz'], b['xyz']), atom_label(a), atom_label(b))
        for a in atoms for b in targets
    )
    return distance, source, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdb', type=Path, default=Path('1CXI.pdb'))
    parser.add_argument('--consurf', type=Path, default=Path('1CXI_A_consurf_grades.txt'))
    parser.add_argument('--chain', default='A')
    parser.add_argument('--ligands', nargs='+', default=['GLC'])
    parser.add_argument('--grade-cutoff', type=int, choices=range(1, 10), default=8)
    parser.add_argument('--ligand-cutoff', type=float, default=6.0)
    parser.add_argument('--calcium-cutoff', type=float, default=5.0)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.ligand_cutoff <= 0 or args.calcium_cutoff <= 0:
        parser.error('Distance cutoffs must be positive')
    scores = read_consurf(args.consurf, args.chain)
    protein, ligands, calcium, bonds = read_pdb(args.pdb, args.chain, set(args.ligands))
    if set(protein) != set(scores):
        raise ValueError(f'PDB/ConSurf residue mismatch: {set(protein) ^ set(scores)}')
    bonded = {key for pair in bonds for key in pair}
    for key in bonded & set(protein):
        if any(a['name'] != 'CYS' for a in protein[key]):
            raise ValueError(f'SSBOND endpoint is not CYS: {key}')
    candidates = []
    for key, atoms in sorted(protein.items()):
        score = scores[key]
        if any(a["name"] != score["residue_name"] for a in atoms):
            raise ValueError(f"PDB/ConSurf identity mismatch: {key}")
        ligand_distance, _, _ = nearest(atoms, ligands)
        calcium_distance, _, _ = nearest(atoms, calcium)
        if (
            score["consurf_grade"] >= args.grade_cutoff
            or ligand_distance < args.ligand_cutoff
            or calcium_distance < args.calcium_cutoff
            or key in bonded
            or key[1] in {183, 370}
        ):
            continue
        candidates.append(f"{key[1]}{key[2]}")
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "mutable_positions.txt"
    output.write_text("".join(position + "\n" for position in candidates), encoding="utf-8")
    print(f"{len(candidates)} mutable residues written to {output}")


if __name__ == "__main__":
    main()
