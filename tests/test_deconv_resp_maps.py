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
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
os.system("export DISPLAY=:0")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import get_data_filename
import numpy as np
import unittest
np.random.seed(seed=42)


class TestDeconvRespMaps(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        self.figure_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

        # rgb convolutional neural network
        model_arch_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.json')
        model_weights_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.h5')
        self.model_arch_file = model_arch_file
        self.model_weights_file = model_weights_file

        # grayscale convolutional neural network
        model_arch_greyscale_file = get_data_filename('data/nn_models/ziletti_diff3d_temp1.json')
        model_weights_greyscale_file = get_data_filename('data/nn_models/ziletti_diff3d_temp1.h5')
        self.model_arch_file_greyscale = model_arch_greyscale_file
        self.model_weights_file_greyscale = model_weights_greyscale_file

    def test_plot_att_response_maps_all_layers_one_img_rgb(self):
        # test the back-projection to image space for all layers with one image for rgb images
        # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
        images = np.random.rand(1, 64, 64, 3)

        # test with one img
        plot_att_response_maps(images, model_arch_file=self.model_arch_file,
                               model_weights_file=self.model_weights_file,
                               figure_dir=self.figure_folder, nb_conv_layers=6,
                               nb_top_feat_maps=4, layer_nb='all', plot_all_filters=False,
                               plot_filter_sum=False,
                               plot_summary=False)

    def test_plot_att_response_maps_all_layers_one_img_greyscale(self):
        # test the back-projection to image space for all layers with one image for grayscale images
        # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
        images = np.random.rand(1, 52, 32, 1)

        # test with one img
        plot_att_response_maps(images, model_arch_file=self.model_arch_file_greyscale,
                               model_weights_file=self.model_weights_file_greyscale,
                               figure_dir=self.figure_folder, nb_conv_layers=6,
                               nb_top_feat_maps=4, layer_nb='all', plot_all_filters=False,
                               plot_filter_sum=False,
                               plot_summary=False)

    def test_plot_att_response_maps_all_layers_multiple_imgs_rgb(self):
        # test the back-projection to image space for all layers with two images
        # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
        images = np.random.rand(2, 64, 64, 3)

        # test with 2 imgs
        plot_att_response_maps(images, model_arch_file=self.model_arch_file,
                               model_weights_file=self.model_weights_file,
                               figure_dir=self.figure_folder, nb_conv_layers=6,
                               nb_top_feat_maps=4, layer_nb='all', plot_all_filters=False,
                               plot_filter_sum=False,
                               plot_summary=False)

    def test_plot_att_response_maps_selected_layers_one_img_rgb(self):
        # test the back-projection to image space for selected layers with one image
        # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
        images = np.random.rand(1, 64, 64, 3)

        # test with one img
        plot_att_response_maps(images, model_arch_file=self.model_arch_file,
                               model_weights_file=self.model_weights_file,
                               figure_dir=self.figure_folder, nb_conv_layers=6,
                               nb_top_feat_maps=4, layer_nb=[0, 3], plot_all_filters=False,
                               plot_filter_sum=False,
                               plot_summary=False)

    def test_plot_att_response_maps_selected_layers_summary_one_img_rgb(self):
        # test the back-projection to image space for selected layers with one image and plot the summary
        # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
        images = np.random.rand(1, 64, 64, 3)

        # test with one img
        plot_att_response_maps(images, model_arch_file=self.model_arch_file,
                               model_weights_file=self.model_weights_file,
                               figure_dir=self.figure_folder, nb_conv_layers=6,
                               nb_top_feat_maps=4, layer_nb=[0, 3], plot_all_filters=False,
                               plot_filter_sum=True,
                               plot_summary=True)

    def tearDown(self):
        # remove the directory after the test
        shutil.rmtree(self.main_folder)
        shutil.rmtree(self.figure_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDeconvRespMaps)
    unittest.TextTestRunner(verbosity=2).run(suite)
