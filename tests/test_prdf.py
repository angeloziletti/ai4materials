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

from ase.build import molecule
from ase.spacegroup import crystal
from ai4materials.descriptors.prdf import PRDF
from ai4materials.descriptors.prdf import get_design_matrix
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import get_data_filename
from ai4materials.wrappers import load_descriptor
import numpy as np
import scipy.sparse
import shutil
import tempfile
import unittest


class TestPRDF(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

        prdf_binaries_desc_file = get_data_filename('/data/descriptors_data/prdf_binaries.tar.gz')
        self.prdf_binaries_desc_file = prdf_binaries_desc_file

    def test_elements_pairs_prdf(self):
        # check the number of unique chemical elements' pairs for the partial radial distribution function
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

        # descriptor calculation
        descriptor = PRDF(configs=self.configs)
        descriptor.calculate(structure)
        prdf = structure.info['descriptor']['prdf']

        n_chem_species = len(set(structure.get_chemical_symbols()))
        self.assertEqual(len(prdf.keys()), n_chem_species*(n_chem_species+1)/2)

    def test_elements_pairs_rdf(self):
        # check that the number of unique chemical elements' pairs for the radial distribution function is one
        # (all chemical species are considered as the same)
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

        # descriptor calculation
        descriptor = PRDF(configs=self.configs, rdf_only=True)
        descriptor.calculate(structure)
        rdf = structure.info['descriptor']['rdf']

        self.assertEqual(len(rdf.keys()), 1)

    def test_molecule(self):
        water = molecule('H2O')
        descriptor = PRDF(configs=self.configs)
        try:
            descriptor.calculate(water)
        except NotImplementedError as err:
            print("{0}".format(err))

    def test_translational_invariance_rdf(self):
        # build crystal structures - one translated w.r.t. the other by an arbitrary vector
        structure = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        structure_translated = structure.copy()
        trans_vector = [3.0, 2.5, 7.0]
        structure_translated.translate(trans_vector)

        # descriptor calculation
        descriptor = PRDF(configs=self.configs, rdf_only=True)
        descriptor.calculate(structure)
        descriptor.calculate(structure_translated)
        rdf = structure.info['descriptor']['rdf']
        rdf_translated = structure_translated.info['descriptor']['rdf']

        # the rdf contains only one unique chemical elements' pair
        # it is defined as '0_0'
        distances = rdf['0_0']['arr']
        distances_translated = rdf_translated['0_0']['arr']

        np.testing.assert_allclose(distances, distances_translated)

    def test_rdf_substitution(self):
        # if chemical identities of atoms are changed the radial distribution fuction must not change
        pass

    def test_write_results(self):
        # test to see if results are correctly written to file
        descriptor = PRDF(configs=self.configs)
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])
        descriptor.calculate(structure)

        # descriptor.write(structure_result, tar=tar, op_id=0)

    def test_get_design_matrix(self):
        # load data with pre-calculated prdfs
        target_list, structure_list = load_descriptor(desc_files=self.prdf_binaries_desc_file, configs=self.configs)

        # calculate design matrix
        design_matrix = get_design_matrix(structure_list, total_bins=50, max_dist=25)

        self.assertIsInstance(design_matrix, scipy.sparse.csr.csr_matrix)
        self.assertTrue(type(design_matrix) is not np.array)
        self.assertEqual(design_matrix.shape[0], len(structure_list))

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPRDF)
    unittest.TextTestRunner(verbosity=2).run(suite)
