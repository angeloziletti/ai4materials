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
from __future__ import division
from __future__ import print_function

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "23/03/18"

import abc
import json
import logging
from ai4materials.utils.utils_config import get_metadata_info
import pandas as pd

logger = logging.getLogger('ai4materials')


class Descriptor(object):
    def __init__(self, configs=None, **params):
        self.name = self.__class__.__name__
        self.configs = configs

        desc_metainfo = get_metadata_info()

        if str(self.name) in desc_metainfo.keys():
            desc_metadata = desc_metainfo[str(self.name)]
            df_metadata = pd.DataFrame(desc_metadata).set_index('name')
            self.desc_metadata = df_metadata
            logger.info('Metadata for descriptor {}: {}'.format(self.name, self.desc_metadata.index.tolist()))
        else:
            logger.debug('No specific descriptor metadata found for descriptor {}.'.format(self.name))

    def __str__(self):
        return self.name + '(' + str(self.__dict__) + ')'

    @abc.abstractmethod
    def calculate(self, structure, **kwargs):
        """Method that calculates the descriptor"""

    @abc.abstractmethod
    def write(self, structure, **kwargs):
        """Method to write the descriptor to file"""

    def write_desc_info(self, desc_info_file, ase_atoms_result):

        with open(desc_info_file, "w") as f:
            f.write("""
    {
        "descriptor_info":[""")

            desc_info = {"descriptor": str(self),
                         "structure_labels": str([item.info['label'] for item in ase_atoms_result])}

            # add descriptor-specific information
            if self.name == 'AtomicFeatures':
                desc_info.update({"selected_feature_list": self.selected_feature_list})

            json.dump(desc_info, f, indent=2)

            f.write("""
        ] }""")
            f.flush()

        return desc_info_file

    def params(self):
        return self.__dict__

    params = staticmethod(params)


def is_descriptor_consistent(structure, descriptor):
    if 'descriptor_name' not in structure.info['descriptor']:
        raise ValueError("No descriptor found. Aborting".format(str(descriptor.name)))

    if 'descriptor_name' in structure.info['descriptor'] and str(descriptor.name) != structure.info['descriptor'][
            'descriptor_name']:
        raise ValueError("Wrong descriptor found. Expecting {}, found {}".format(str(descriptor.name),
                                                                                 structure.info['descriptor'][
                                                                                     'descriptor_name']))
    return True
