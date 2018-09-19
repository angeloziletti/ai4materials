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

import logging
import unittest
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger


class TestUtilsConfig(unittest.TestCase):
    def setUp(self):
        pass

    def test_read_configs_empty_file(self):
        configs = set_configs()

        self.assertIsInstance(configs, dict)

    def test_setup_logger(self):
        logger = setup_logger(configs=None, level=None, display_configs=False)

        self.assertIsInstance(logger, logging.Logger)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtilsConfig)
    unittest.TextTestRunner(verbosity=2).run(suite)
