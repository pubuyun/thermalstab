"""Independent checks of generated results against the supplied 1CXI files."""

import unittest
from collections import defaultdict
from pathlib import Path

from filter_mutable_residues import nearest, read_consurf, read_pdb


ROOT = Path(__file__).resolve().parents[1]


class ResidueFilterTests(unittest.TestCase):
    def test_all_distances_and_selection_independently(self):
        # Whitespace parsing and squared-distance arithmetic are independent
        # of the production fixed-column parser and math.dist implementation.
        protein = defaultdict(list)
        ligands, calcium = [], []
        for line in (ROOT / '1CXI.pdb').read_text().splitlines():
            fields = line.split()
            if not fields or fields[0] not in {'ATOM', 'HETATM'}:
                continue
            # Occupancy and B-factor can touch (e.g. 1.00100.00).
            if fields[-1] in {'H', 'D', 'T'} or float(line[54:60]) <= 0:
                continue
            xyz = tuple(map(float, fields[6:9]))
            if fields[0] == 'ATOM' and fields[4] == 'A':
                protein[int(fields[5])].append(xyz)
            elif fields[3] == 'GLC':
                ligands.append(xyz)
            elif fields[3] == 'CA':
                calcium.append(xyz)
        scores = read_consurf(ROOT / '1CXI_A_consurf_grades.txt', 'A')
        self.assertEqual(len(scores), len(protein))
        self.assertEqual(len(ligands), 69)
        self.assertEqual(len(calcium), 2)
        expected_positions = []
        for position in sorted(protein):
            distances = []
            for targets in (ligands, calcium):
                squared = min(sum((x-y)**2 for x, y in zip(a, b)) for a in protein[position] for b in targets)
                distances.append(squared)
            if scores[('A', position, '')]['consurf_grade'] < 8 and distances[0] >= 36 and distances[1] >= 25 and position not in {43, 50, 183, 370}:
                expected_positions.append(str(position))
        candidates = (ROOT / 'results/mutable_positions.txt').read_text(encoding='utf-8').splitlines()
        self.assertEqual(candidates, expected_positions)
        self.assertEqual(len(candidates), 451)
        self.assertNotIn('183', candidates)
        self.assertNotIn('370', candidates)

    def test_mapping_and_disulfide(self):
        scores = read_consurf(ROOT / '1CXI_A_consurf_grades.txt', 'A')
        protein, _, _, bonds = read_pdb(ROOT / '1CXI.pdb', 'A', {'GLC'})
        self.assertEqual(set(scores), set(protein))
        self.assertEqual(len(scores), 686)
        self.assertEqual(bonds, [(('A', 43, ''), ('A', 50, ''))])
        for key in bonds[0]:
            self.assertEqual(scores[key]['wild_type'], 'C')
        # A non-bonded Cys must remain eligible if other criteria pass.
        self.assertEqual(scores[('A', 400, '')]['wild_type'], 'C')

    def test_nearest_uses_sidechain_and_all_targets(self):
        def atom(name, xyz):
            return dict(key=('A', 1, ''), name='LYS', atom=name, altloc='', xyz=xyz)
        distance, source, _ = nearest(
            [atom('CA', (0, 0, 0)), atom('NZ', (10, 0, 0))],
            [atom('O1', (30, 0, 0)), atom('O2', (14, 0, 0))],
        )
        self.assertEqual(distance, 4)
        self.assertTrue(source.endswith(':NZ'))


if __name__ == '__main__':
    unittest.main()
