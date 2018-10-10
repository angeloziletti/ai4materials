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

from functools import partial
from ai4materials.utils.utils_config import set_configs
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
from ai4materials.dataprocessing.preprocessing import make_data_sets
from ai4materials.models.cnn_architectures import cnn_nature_comm_ziletti2018
from ai4materials.models.cnn_nature_comm_ziletti2018 import predict
from ai4materials.models.cnn_nature_comm_ziletti2018 import train_neural_network
import numpy as np
import random
import tempfile
import unittest
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing


# @unittest.skip("temporarily disabled")
class TestCnnNatureCommZiletti2018(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

        dataset_folder = '/home/ziletti/Documents/calc_nomadml/2d_nature_comm/datasets_2d/'
        self.dataset_folder = dataset_folder

        n_samples = 100

        crystal_classes = ['bct139', 'bct142', 'rh/hex', 'sc', 'fcc', 'diam', 'bcc']
        text_labels = np.array([random.choice(crystal_classes) for _ in range(n_samples)])
        x_pristine = np.random.rand(n_samples, 64, 64, 3)
        x_vac25 = np.random.rand(n_samples, 64, 64, 3)

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(text_labels)
        y_pristine = label_encoder.transform(text_labels)
        # the true labels are the same for pristine and defectives
        y_vac25 = y_pristine

        self.x_pristine = x_pristine
        self.y_pristine = y_pristine
        self.text_labels = text_labels

        self.x_vac25 = x_vac25
        self.y_vac25 = y_vac25

    def test_train_neural_network(self):
        dataset = make_data_sets(x_train_val=self.x_pristine, y_train_val=self.y_pristine, split_train_val=True,
                                 test_size=0.1, x_test=self.x_vac25, y_test=self.y_vac25)

        # load the data
        x_train = dataset.train.images
        y_train = dataset.train.labels

        partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                             kernel_sizes=[3, 3, 3, 3, 3, 3], max_pool_strides=[2, 2],
                                             hidden_layer_size=128)

        # use x_train also for validation - this is only to run the test
        results = train_neural_network(x_train=x_train, y_train=y_train, x_val=x_train, y_val=y_train,
                                       configs=self.configs, partial_model_architecture=partial_model_architecture,
                                       nb_epoch=1)

    def test_predict(self):
        dataset = make_data_sets(x_train_val=self.x_pristine, y_train_val=self.y_pristine, split_train_val=False,
                                 x_test=self.x_vac25, y_test=self.y_vac25)

        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.text_labels)
        numerical_labels = label_encoder.transform(self.text_labels)

        # load the data
        x_test = dataset.test.images
        y_test = dataset.test.labels

        results = predict(x_test, y_test, configs=self.configs,
                          numerical_labels=numerical_labels, text_labels=self.text_labels)

        self.assertIsInstance(results, dict)

        # probabilities should be between 0. and 1.
        self.assertLessEqual(np.amax(results['prob_predictions']), 1.0)
        self.assertGreaterEqual(np.amin(results['prob_predictions']), 0.0)

        # target_pred_class unique values should be at most 7
        self.assertLessEqual(len(set(results['target_pred_class'])), 7)
        self.assertLessEqual(np.amax(results['target_pred_class']), 7)
        self.assertGreaterEqual(np.amax(results['prob_predictions']), 0)

        # confusion matrix should be a numpy array
        self.assertIsInstance(results['confusion_matrix'], np.ndarray)

        # string_probs are a list of strings - one element for each prediction
        self.assertIsInstance(results['string_probs'], list)

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCnnNatureCommZiletti2018)
    unittest.TextTestRunner(verbosity=2).run(suite)
