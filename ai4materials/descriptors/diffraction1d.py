#!/usr/bin/python
# coding=utf-8
# Copyright 2016-2019 Angelo Ziletti
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
__date__ = "13/03/18"

import logging
from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
import numpy as np
import os
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
logger = logging.getLogger('ai4materials')


class Diffraction1D(Descriptor):
    def __init__(self, configs, wavelength="CuKa"):
        super(Diffraction1D, self).__init__(configs=configs)

        self.wavelength = wavelength

    def calculate(self, structure, show=False):

        c = XRDCalculator(wavelength=self.wavelength)
        xrd_pattern = c.get_xrd_pattern(AseAtomsAdaptor.get_structure(structure))

        if show:
            c.show_xrd_plot(AseAtomsAdaptor.get_structure(structure))

        descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), xrd_pattern=xrd_pattern)

        structure.info['descriptor'] = descriptor_data

        return structure

    def write(self, structure, tar, write_xrd_pattern=True):

        desc_folder = self.configs['io']['desc_folder']

        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')

        if write_xrd_pattern:
            xrd_pattern = structure.info['descriptor']['xrd_pattern']
            xrd_pattern_filename_npy = os.path.abspath(
                os.path.normpath(os.path.join(desc_folder, structure.info['label'] + self.desc_metadata.ix['xrd_pattern']['file_ending'])))
            np.save(xrd_pattern_filename_npy, xrd_pattern)
            structure.info['xrd_pattern_filename_npy'] = xrd_pattern_filename_npy
            tar.add(structure.info['xrd_pattern_filename_npy'])
