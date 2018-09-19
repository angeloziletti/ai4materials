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

import ase
from ase.spacegroup import crystal
from ai4materials.descriptors.diffraction3d import DISH
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_crystals import create_supercell
import numpy as np
import random
import tempfile
import unittest
import shutil


# @unittest.skip("temporarily disabled")
class TestDISH(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

    def test_normalization(self):
        descriptor = DISH(configs=self.configs)

        # build crystal structure
        bcc_fe = crystal('Fe', [(0, 0, 0)], spacegroup=229, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        structure = create_supercell(bcc_fe, target_nb_atoms=256)

        # calculate the descriptor
        descriptor.calculate(structure)
        intensity = structure.info['descriptor']['diffraction_3d_sh_spectrum']

        # check if intensity_rgb is normalized
        self.assertAlmostEqual(np.amax(intensity), 1.0)
        self.assertAlmostEqual(np.amin(intensity), 0.0)

    def test_molecule(self):
        # build molecule
        water = ase.build.molecule('H2O')
        # it should raise NotImplementedError

    def test_approximate_size_invariance(self):
        descriptor = DISH(configs=self.configs)

        # build crystal structures - two supercell sizes
        fcc_al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        structure_128 = create_supercell(fcc_al, target_nb_atoms=128)
        structure_256 = create_supercell(fcc_al, target_nb_atoms=256)

        # calculate the descriptor
        descriptor.calculate(structure_128)
        descriptor.calculate(structure_256)
        intensity_128 = structure_128.info['descriptor']['diffraction_3d_sh_spectrum']
        intensity_256 = structure_256.info['descriptor']['diffraction_3d_sh_spectrum']

        # test if the two diffraction fingerprints are different by less than 20% (20% is arbitrary)
        self.assertLessEqual(np.abs(np.amax(intensity_256-intensity_128)), 0.20)

    @unittest.skip("temporarily disabled")
    def test_approximate_rotational_invariance(self):
        descriptor = DISH(configs=self.configs)

        # build crystal structures - two different random rotations
        fcc_al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        fcc_all_supercell_rot1 = create_supercell(fcc_al, target_nb_atoms=128)
        fcc_all_supercell_rot2 = create_supercell(fcc_al, target_nb_atoms=128)

        # rotate 1st structure
        alphas = (random.random() * 360.0, random.random() * 360.0, random.random() * 360.0)
        fcc_all_supercell_rot1.rotate(alphas[0], 'x', rotate_cell=True, center='COU')
        fcc_all_supercell_rot1.rotate(alphas[1], 'y', rotate_cell=True, center='COU')
        fcc_all_supercell_rot1.rotate(alphas[2], 'z', rotate_cell=True, center='COU')

        # rotate 2nd structure
        alphas = (random.random() * 360.0, random.random() * 360.0, random.random() * 360.0)
        fcc_all_supercell_rot2.rotate(alphas[0], 'x', rotate_cell=True, center='COU')
        fcc_all_supercell_rot2.rotate(alphas[1], 'y', rotate_cell=True, center='COU')
        fcc_all_supercell_rot2.rotate(alphas[2], 'z', rotate_cell=True, center='COU')

        # calculate the descriptor
        descriptor.calculate(fcc_all_supercell_rot1)
        descriptor.calculate(fcc_all_supercell_rot2)
        spectrum_rot1 = fcc_all_supercell_rot1.info['descriptor']['diffraction_3d_sh_spectrum']
        spectrum_rot2 = fcc_all_supercell_rot2.info['descriptor']['diffraction_3d_sh_spectrum']

        # test if the two diffraction fingerprints are different by less than 20% (20% is arbitrary)
        self.assertLessEqual(np.abs(np.amax(spectrum_rot1-spectrum_rot2)), 0.20)


    def test_approximate_translational_invariance(self):
        # build crystal structures - one translated w.r.t. the other by an arbitrary vector
        structure = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
        trans_vector = [1.0, -2.0, 5.0]
        structure_translated = structure.copy()
        structure_translated.translate(trans_vector)
        structure = create_supercell(structure, target_nb_atoms=128)
        structure_translated = create_supercell(structure_translated, target_nb_atoms=128)

        # calculate the descriptor
        descriptor = DISH(configs=self.configs)
        descriptor.calculate(structure)
        descriptor.calculate(structure_translated)
        intensity = structure.info['descriptor']['diffraction_3d_sh_spectrum']
        intensity_translated = structure_translated.info['descriptor']['diffraction_3d_sh_spectrum']

        # test if the two diffraction fingerprints are different by less than 20% (20% is arbitrary)
        np.testing.assert_allclose(intensity, intensity_translated, 0.2)

    def test_binaries(self):
        # check if the diffraction fingerprint calculation works for more than one element
        descriptor = DISH(configs=self.configs)

        # build binary structure
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

        descriptor.calculate(structure)

    def test_write_results(self):
        # test to see if results are correctly written to file
        descriptor = DISH(configs=self.configs)
        structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                            cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

        descriptor.calculate(structure)

        # descriptor.write(structure_result, tar=tar, op_id=0)

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDISH)
    unittest.TextTestRunner(verbosity=2).run(suite)
