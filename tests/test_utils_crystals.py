#!/usr/bin/python
# coding=utf-8
# Copyright 2016-2018 Angelo Ziletti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/08/17"

import ai4materials.utils.utils_crystals
import ase
import unittest
import random
random.seed(42)


class TestUtilsCrystals(unittest.TestCase):
    def setUp(self):
        bulk_structure = ase.build.bulk('Cu', 'fcc', a=3.6, orthorhombic=True)
        molecule = ase.build.molecule('H2O')
        self.bulk_structure = bulk_structure
        self.molecule = molecule

    def test_create_supercell(self):
        supercell_bulk = ai4materials.utils.utils_crystals.create_supercell(self.bulk_structure,
                                                                            create_replicas_by='nb_atoms',
                                                                            target_nb_atoms=128)

        self.assertIsInstance(supercell_bulk, ase.Atoms)
        self.assertGreaterEqual(len(supercell_bulk), len(self.bulk_structure))
        self.assertItemsEqual(supercell_bulk.get_pbc(), self.bulk_structure.get_pbc())

        # supercell_molecule = ai4materials.utils.utils_crystals.create_supercell(self.molecule, create_replicas_by='nb_atoms', target_nb_atoms=128)  #  # self.assertIsInstance(supercell_molecule, ase.Atoms)  # self.assertGreaterEqual(len(supercell_molecule), len(self.molecule))  # self.assertItemsEqual(supercell_molecule.get_pbc(), self.molecule.get_pbc())

    def test_convert_energy_substance(self):
        item_1 = random.random()
        item_2 = ai4materials.utils.utils_crystals.convert_energy_substance('J', item_1, energy_unit='eV')
        item_3 = ai4materials.utils.utils_crystals.convert_energy_substance('eV', item_2, energy_unit='J')

        self.assertEqual(item_1, item_3)

        item_1 = random.random()
        item_2 = ai4materials.utils.utils_crystals.convert_energy_substance('J', item_1, energy_unit='eV')
        item_3 = ai4materials.utils.utils_crystals.convert_energy_substance('eV', item_2, energy_unit='J')

        self.assertEqual(item_1, item_3)

    def test_get_nn_distance(self):
        # check if we get the same scale factor if a supercell is built
        scale_factor = ai4materials.utils.utils_crystals.get_nn_distance(atoms=self.bulk_structure,
                                                                         distribution='quantile_nn')
        scale_factor_supercell = ai4materials.utils.utils_crystals.get_nn_distance(atoms=self.bulk_structure * 3,
                                                                                   distribution='quantile_nn')

        self.assertEqual(scale_factor, scale_factor_supercell)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtilsCrystals)
    unittest.TextTestRunner(verbosity=2).run(suite)
