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
from ai4materials.models.cnn_architectures import cnn_nature_comm_ziletti2018
from ai4materials.models.cnn_nature_comm_ziletti2018 import load_datasets
from ai4materials.models.cnn_nature_comm_ziletti2018 import predict
from ai4materials.models.cnn_nature_comm_ziletti2018 import train_neural_network
from ai4materials.utils.utils_config import setup_logger
import numpy as np
import shutil
import os


if __name__ == '__main__':
    configs = set_configs()
    logger = setup_logger(configs, level='DEBUG', display_configs=False)

    # dataset_folder = '/home/ziletti/Documents/calc_nomadml/2d_nature_comm/datasets_2d_downloaded/'

    dataset_folder = configs['io']['main_folder']
    x_pristine, y_pristine, dataset_info_pristine, x_vac25, y_vac25, dataset_info_vac25 = load_datasets(dataset_folder)

    train_set_name = 'pristine_dataset'
    path_to_x_pristine = os.path.join(dataset_folder, train_set_name + '_x.pkl')
    path_to_y_pristine = os.path.join(dataset_folder, train_set_name + '_y.pkl')
    path_to_summary_pristine = os.path.join(dataset_folder, train_set_name + '_summary.json')

    test_set_name = 'vac25_dataset'
    path_to_x_vac25 = os.path.join(dataset_folder, test_set_name + '_x.pkl')
    path_to_y_vac25 = os.path.join(dataset_folder, test_set_name + '_y.pkl')
    path_to_summary_vac25 = os.path.join(dataset_folder, test_set_name + '_summary.json')

    x_pristine, y_pristine, dataset_info_pristine = load_dataset_from_file(path_to_x_pristine, path_to_y_pristine,
                                                                           path_to_summary_pristine)

    x_vac25, y_vac25, dataset_info_vac25 = load_dataset_from_file(path_to_x_vac25, path_to_y_vac25,
                                                                  path_to_summary_vac25)

    # partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[32, 32, 16, 16, 8, 8],
    partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[2, 2, 2, 2, 2, 2],
                                         kernel_sizes=[3, 3, 3, 3, 3, 3], max_pool_strides=[2, 2],
                                         hidden_layer_size=128)

    # # use x_train also for validation - this is only to run the test
    # results = train_neural_network(x_train=x_pristine, y_train=y_pristine, x_val=x_pristine, y_val=y_pristine,
    #                                configs=configs, partial_model_architecture=partial_model_architecture,
    #                                nb_epoch=1)

    text_labels = np.asarray(dataset_info_vac25["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_vac25["data"][0]["numerical_labels"])

    results = predict(x_vac25, y_vac25, configs=configs, numerical_labels=numerical_labels,
                      text_labels=text_labels, model=None)

    # Image(filename=conf_matrix_file, width=800)

    # remove tmp folder
    # shutil.rmtree(main_folder)

