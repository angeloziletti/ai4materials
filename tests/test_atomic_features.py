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

import shutil
import tempfile

# atomic_data_dir = os.path.abspath(os.path.normpath("/home/ziletti/nomad/nomad-lab-base/analysis-tools/atomic-data"))
# sys.path.insert(0, atomic_data_dir)

from ase.spacegroup import crystal
from ai4materials.descriptors.atomic_features import AtomicFeatures
from ai4materials.descriptors.atomic_features import get_table_atomic_features
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_crystals import create_supercell
import pandas as pd
import unittest

@unittest.skip("temporarily disabled")
class TestAtomicFeatures(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

    def test_features_dataframe(self):
        # test if atomic features are read and returned as panda dataframe
        # build structure
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

        # define descriptor
        selected_feature_list = ['atomic_ionization_potential', 'atomic_electron_affinity', 'atomic_rs_max',
                                 'atomic_rp_max', 'atomic_rd_max']
        kwargs = {'feature_order_by': 'atomic_mulliken_electronegativity', 'energy_unit': 'eV',
                  'length_unit': 'angstrom'}
        descriptor = AtomicFeatures(configs=self.configs, **kwargs)

        # calculate descriptor
        descriptor.calculate(structure, selected_feature_list=selected_feature_list)
        df_atomic_features = structure.info['descriptor']['atomic_features_table']

        self.assertIsInstance(df_atomic_features, pd.DataFrame)
        # the columns in the atomic feature dataframe should be #selected_feature_list*2 + 1 (the index)
        self.assertEqual(len(df_atomic_features.columns.tolist()), len(selected_feature_list)*2+1)

    def test_elem_solids(self):
        # build structure
        structure = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])

        # define descriptor
        selected_feature_list = ['atomic_ionization_potential', 'atomic_electron_affinity', 'atomic_rs_max']
        kwargs = {'feature_order_by': 'atomic_mulliken_electronegativity', 'energy_unit': 'eV',
                  'length_unit': 'angstrom'}
        descriptor = AtomicFeatures(configs=self.configs, **kwargs)

        descriptor.calculate(structure, selected_feature_list=selected_feature_list)
        df_atomic_features = structure.info['descriptor']['atomic_features_table']

        self.assertIsInstance(df_atomic_features, pd.DataFrame)
        # the columns in the atomic feature dataframe should be #selected_feature_list*2 + 1 (the index)
        # this is because also elemental solids are treated as binaries, where the two chemical species are the same
        self.assertEqual(len(df_atomic_features.columns.tolist()), len(selected_feature_list)*2+1)

    def test_get_table_atomic_features(self):
        # build structures
        fcc_al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        bcc_fe = crystal('Fe', [(0, 0, 0)], spacegroup=229, cellpar=[2.87, 2.87, 2.87, 90, 90, 90])
        diamond_c = crystal('C', [(0, 0, 0)], spacegroup=227, cellpar=[3.57, 3.57, 3.57, 90, 90, 90])
        hcp_mg = crystal('Mg', [(1. / 3., 2. / 3., 3. / 4.)], spacegroup=194, cellpar=[3.21, 3.21, 5.21, 90, 90, 120])
        structures = [fcc_al, bcc_fe, diamond_c, hcp_mg]

        # define descriptor
        selected_feature_list = ['atomic_ionization_potential', 'atomic_electron_affinity', 'atomic_rs_max']
        kwargs = {'feature_order_by': 'atomic_mulliken_electronegativity', 'energy_unit': 'eV',
                  'length_unit': 'angstrom'}
        descriptor = AtomicFeatures(configs=self.configs, **kwargs)

        # calculate descriptor
        structure_results = []
        for structure in structures:
            structure_results.append(descriptor.calculate(structure, selected_feature_list=selected_feature_list))

        df_atomic_features = get_table_atomic_features(structure_results)

        self.assertIsInstance(df_atomic_features, pd.DataFrame)
        self.assertEqual(len(df_atomic_features.columns.tolist()), len(selected_feature_list)*2+1)
        self.assertEqual(len(df_atomic_features), len(structures))

    def test_size_invariance(self):
        # replicating a unit cell must not change the atomic features
        structure = crystal('C', [(0, 0, 0)], spacegroup=227, cellpar=[3.57, 3.57, 3.57, 90, 90, 90])
        structure_supercell = create_supercell(structure, target_nb_atoms=128)

        # define descriptor
        selected_feature_list = ['atomic_ionization_potential', 'atomic_electron_affinity', 'atomic_rs_max']
        kwargs = {'feature_order_by': 'atomic_mulliken_electronegativity', 'energy_unit': 'eV',
                  'length_unit': 'angstrom'}
        descriptor = AtomicFeatures(configs=self.configs, **kwargs)

        # calculate descriptor
        descriptor.calculate(structure, selected_feature_list=selected_feature_list)
        df_atomic_features = structure.info['descriptor']['atomic_features_table']
        descriptor.calculate(structure_supercell, selected_feature_list=selected_feature_list)
        df_atomic_features_supercell = structure_supercell.info['descriptor']['atomic_features_table']

        self.assertIsInstance(df_atomic_features, pd.DataFrame)
        self.assertIsInstance(df_atomic_features_supercell, pd.DataFrame)
        # check if the two dataframe are the same
        # self.assertEqual(df_atomic_features, df_atomic_features_supercell)

    def tearDown(self):
        # remove the directory after the test
        shutil.rmtree(self.main_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAtomicFeatures)
    unittest.TextTestRunner(verbosity=2).run(suite)
