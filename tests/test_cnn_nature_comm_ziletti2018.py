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
from ai4materials.utils.utils_config import get_data_filename
import numpy as np
import tempfile
import unittest
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
import os


@unittest.skip("temporarily disabled")
class TestCnnNatureCommZiletti2018(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

        dataset_folder = '/home/ziletti/Documents/calc_nomadml/2d_nature_comm/datasets_2d/'
        self.dataset_folder = dataset_folder

        train_set_name = 'pristine_dataset'
        path_to_x = os.path.join(dataset_folder, train_set_name + '_x.pkl')
        path_to_y = os.path.join(dataset_folder, train_set_name + '_y.pkl')
        path_to_summary = os.path.join(dataset_folder, train_set_name + '_summary.json')

        x_pristine, y_pristine, dataset_info_pristine = load_dataset_from_file(path_to_x, path_to_y, path_to_summary)

        test_set_name = 'vac0.25_dataset'
        path_to_x = os.path.join(dataset_folder, test_set_name + '_x.pkl')
        path_to_y = os.path.join(dataset_folder, test_set_name + '_y.pkl')
        path_to_summary = os.path.join(dataset_folder, test_set_name + '_summary.json')

        x_vac25, y_vac25, dataset_info_vac25 = load_dataset_from_file(path_to_x, path_to_y, path_to_summary)

        self.x_pristine = x_pristine
        self.y_pristine = y_pristine
        self.dataset_info_pristine = dataset_info_pristine

        self.x_vac25 = x_vac25
        self.y_vac25 = y_vac25
        self.dataset_info_vac25 = dataset_info_vac25

    def test_train_neural_network(self):
        dataset = make_data_sets(x_train_val=self.x_pristine, y_train_val=self.y_pristine, split_train_val=True,
                                 test_size=0.1, x_test=self.x_vac25, y_test=self.y_vac25)

        # load the data
        x_train = dataset.train.images
        y_train = dataset.train.labels

        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.02, random_state=0)
        for index_0, index_1 in sss.split(x_train, y_train):
            _, x_train_sub = x_train[index_0], x_train[index_1]
            _, y_train_sub = y_train[index_0], y_train[index_1]

        partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                             kernel_sizes=[3, 3, 3, 3, 3, 3], max_pool_strides=[2, 2],
                                             hidden_layer_size=128)

        # use x_train also for validation - this is only to run the test
        results = train_neural_network(x_train=x_train_sub, y_train=y_train_sub, x_val=x_train_sub, y_val=y_train_sub,
                                       configs=self.configs, partial_model_architecture=partial_model_architecture,
                                       nb_epoch=1)

    # @unittest.skip("temporarily disabled")
    def test_predict(self):
        dataset = make_data_sets(x_train_val=self.x_pristine, y_train_val=self.y_pristine, split_train_val=False,
                                 x_test=self.x_vac25, y_test=self.y_vac25)

        text_labels = np.asarray(self.dataset_info_vac25["data"][0]["text_labels"])
        numerical_labels = np.asarray(self.dataset_info_vac25["data"][0]["numerical_labels"])

        # load the data
        x_test = dataset.test.images
        y_test = dataset.test.labels

        # select only 1% of the data to predict faster - can be omitted
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.01, random_state=0)
        for index_0, index_1 in sss.split(x_test, y_test):
            _, x_test_sub = x_test[index_0], x_test[index_1]
            _, y_test_sub = y_test[index_0], y_test[index_1]
            _, text_labels_sub = text_labels[index_0], text_labels[index_1]
            _, numerical_labels_sub = numerical_labels[index_0], numerical_labels[index_1]

        results = predict(x_test_sub, y_test_sub, configs=self.configs,
                          numerical_labels=numerical_labels_sub, text_labels=text_labels_sub)

        self.assertIsInstance(results, dict)

        # probabilities should be between 0. and 1.
        self.assertLessEqual(np.amax(results['prob_predictions']), 1.0)
        self.assertGreaterEqual(np.amin(results['prob_predictions']), 0.0)

        # target_pred_class unique values should be at most 7 (that can be less because we use only 2% of the dataset)
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
