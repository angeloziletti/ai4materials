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


class TestWrappers(unittest.TestCase):
    def setUp(self):
        bulk_structure = ase.build.bulk('Cu', 'fcc', a=3.6, orthorhombic=True)
        molecule = ase.build.molecule('H2O')
        self.bulk_structure = bulk_structure
        self.molecule = molecule

    def test_calc_descriptor(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWrappers)
    unittest.TextTestRunner(verbosity=2).run(suite)
