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
__date__ = "23/09/18"

import os
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend as K
from keras.optimizers import Adam
from keras.models import model_from_json
import logging
import numpy as np
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# os.system("export DISPLAY=:0")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.cm as cm
from matplotlib import gridspec
from ai4materials.utils.utils_plotting import make_multiple_image_plot
from ai4materials.utils.utils_plotting import rgb_colormaps
from numpy import amin, amax
import os
import os.path
import time
from PIL import Image

logger = logging.getLogger('ai4materials')


def load_model(model_arch_file, model_weights_file):
    """ Load Keras model from .json and .h5 files

    """

    json_file = open(model_arch_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    logger.info("Loading model weights.")
    model.load_weights(model_weights_file)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    logger.debug("Model summary below.")

    return model


def plot_att_response_maps(data, model_arch_file, model_weights_file,
                           figure_dir, nb_conv_layers, layer_nb='all',
                           nb_top_feat_maps=4, filename_maps="attentive_response_maps",
                           cmap=cm.hot,
                           plot_all_filters=False,
                           plot_filter_sum=True, plot_summary=True):
    """ Plot attentive response maps given a Keras trained model and
    input images.

    Parameters:

    data: ndarray, shape (n_images, dim1, dim2, channels)
        Array of input images that will be used to calculate the
        attentive response maps.

    model_arch_file: string
        Full path to the model architecture file (.json format) written
        by Keras after the neural network training.
        This is used by the load_model function to load the neural network
        architecture.

    model_weights_file: string
        Full path to the model weights file (.h5 format) written by Keras
        after the neural network training .
        This is used by the load_model function to load the neural network
        architecture.

    figure_dir: string
        Full path of the directory where the images resulting from the
        transposed convolution procedure will be saved.

    nb_conv_layers: int
        Numbers of Convolution2D layers in the neural network architecture.

    layer_nb: list of int, or 'all'
        List with the layer number which will be deconvolved starting from 0.
        E.g. layer_nb=[0, 1, 4] will deconvolve the 1st, 2nd, and 5th
        convolution2d layer. Only up to 6 conv_2d layers are supported.
        If 'all' is selected, all conv_2d layers will be deconvolved,
        up to nb_conv_layers.

    nb_top_feat_maps: int
        Number of the top attentive response maps to be calculated and
        plotted. It must be <= to the minimum number of filters used in the
        neural network layers. This is not checked by the code, and
        respecting this criterion is up to the user.

    filename_maps: str
        Base filename (without extension and path) of the files where the
        attentive response maps will be saved.

    cmap: Matplotlib cmap, optional, default=`cm.hot`
        Type of coloring for the heatmap, if images are greyscale.
        Possible cmaps can be found here:
        https://matplotlib.org/examples/color/colormaps_reference.html
        If images are RGB, then an RGB color map is used.
        The RGB colormap can be found at :py:mod:`ai4materials.utils.utils_plotting.rgb_colormaps`.

    plot_all_filters: bool
        If True, plot and save the nb_top_feat_maps for each layer.
        The files will be saved in different folders according to the layer:
        - "convolution2d_1" for the 1st layer
        - "convolution2d_2" for the 2nd layer
        etc.

    plot_filter_sum: bool
        If True, plot and save the sum of all the filters for a given layer.

    plot_summary: bool
        If True, plot and save a summary figure containing:
        (left) input image
        (center) nb_top_feat_maps filters for each deconvolved layer
        (right) sum of the all filters of the last layer
        If set to True, also plot_filter_sum must be set to True.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    model = load_model(model_arch_file, model_weights_file)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    nb_input_imgs = data.shape[0]
    # img_size = (#px #py #channels)
    img_size = (data.shape[1], data.shape[2], data.shape[3])

    all_target_layers = []
    for idx_conv_layer in range(nb_conv_layers):
        all_target_layers.append('convolution2d_' + str(idx_conv_layer + 1))

    # plot the attentive response maps for all images in the dataset
    for idx_img in range(nb_input_imgs):
        plt.clf()
        logger.info("Calculating attentive response map for figure {0}/{1}".format(idx_img + 1, nb_input_imgs))
        # plot input picture
        input_picture = data[idx_img]
        input_picture = input_picture.reshape(img_size[2], img_size[0], img_size[1])
        plt.axis('off')
        # plt.imshow(input_picture, cmap='gray')

        # save input picture
        filename_input = str(idx_img) + "_img_input.png"
        filename_input_full = os.path.abspath(os.path.normpath(os.path.join(figure_dir, filename_input)))
        # to avoid whitespaces when saving
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(filename_input_full, dpi=100, bbox_inches='tight', pad_inches=0.0, format='png')

        # save as proper picture
        rgb_array = np.zeros((img_size[0], img_size[1], img_size[2]), 'uint8')
        if img_size[2] == 3:
            rgb_array[..., 0] = input_picture[0] * 255
            rgb_array[..., 1] = input_picture[1] * 255
            rgb_array[..., 2] = input_picture[2] * 255
            img = Image.fromarray(rgb_array)
        elif img_size[2] == 1:
            img_array = np.array(input_picture[0] * 255).reshape((input_picture.shape[1], input_picture.shape[2])).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        else:
            raise Exception("Unexpected number of color channels: {}".format(img_size[2]))

        img.save(filename_input_full)

        # get one image at a time to deconvolve
        input_data = data[idx_img]
        input_data = input_data.reshape(-1, img_size[2], img_size[0], img_size[1])
        nb_channels = img_size[2]

        # define colormaps
        cmaps = []
        if nb_channels == 3:
            # from 0 to full red, green and blue
            colors = ["red", "green", "blue"]
            for color in colors:
                cmaps.append(rgb_colormaps(color))
        elif nb_channels == 1:
            cmaps.append(cmap)
        else:
            raise Exception("Unexpected number of color channels: {}".format(img_size[2]))

        if layer_nb == 'all':
            target_layers = all_target_layers
        else:
            target_layers = [all_target_layers[i] for i in list(layer_nb)]

        output_layers = []
        for target_layer in target_layers:
            # output.shape = #img, #filters, #channels, #px, #py
            logger.info("Processing layer: {0}".format(target_layer))
            # here the filters are calculated by transposed convolution
            output, feat_maps = deconv_visualize(model, target_layer, input_data, nb_top_feat_maps)
            output_layers.append(output)

        output_layers = np.asarray(output_layers)
        # output_for_maps.shape = #img x #filters, #channels, #px, #py
        output_for_maps = np.reshape(output_layers, (-1, img_size[2], img_size[0], img_size[1]))

        filename_maps_i = filename_maps + "_img_" + str(idx_img) + ".png"
        filename_maps_i_full = os.path.abspath(os.path.normpath(os.path.join(figure_dir, filename_maps_i)))
        # plot images
        filenames_ch = make_multiple_image_plot(output_for_maps, title="Attentive response maps",
                                                n_rows=nb_top_feat_maps, n_cols=len(target_layers),
                                                cmap=cmap,
                                                vmin=None, vmax=None, save=True, filename=filename_maps_i_full)

        # processing one image at a time
        # output_for_filters.shape = #layers #filters #channels #px, #py
        output_for_filters = np.reshape(
            output_layers,
            (len(target_layers),
             nb_top_feat_maps,
             img_size[2],
             img_size[0],
             img_size[1]))

        if plot_all_filters:
            for idx_layer, target_layer in enumerate(target_layers):
                # make a dir for each target layer
                layer_dir = os.path.abspath(os.path.normpath(os.path.join(figure_dir, target_layer)))
                if not os.path.exists(layer_dir):
                    os.makedirs(layer_dir)

                for idx_filter in range(nb_top_feat_maps):
                    filename_filter_rgb_list = []
                    for idx_ch in range(nb_channels):
                        plt.clf()
                        data_filter = output_for_filters[idx_layer, idx_filter, idx_ch, :, :]

                        # show only positive filters
                        vmin = max(0.0, amin(data_filter))
                        vmax = amax(data_filter)
                        plt.axis('off')
                        plt.imshow(data_filter, cmap=cmaps[idx_ch], vmin=vmin, vmax=vmax * 1.0)

                        # to avoid whitespaces when saving
                        plt.gca().xaxis.set_major_locator(plt.NullLocator())
                        plt.gca().yaxis.set_major_locator(plt.NullLocator())

                        filter_name = str(target_layer) + "_" + "filter_nb" + str(idx_filter) + \
                            "_image_" + str(idx_img) + "_ch" + str(idx_ch) + ".png"
                        filename_filter = os.path.abspath(os.path.normpath(os.path.join(layer_dir, filter_name)))
                        plt.savefig(filename_filter, dpi=100, bbox_inches='tight', pad_inches=0.0, format='png')

                        # append to calculate the overlayed rgb picture
                        filename_filter_rgb_list.append(filename_filter)

                    # plot the rgb sum of all filters
                    # get dimensions of first image (assuming all images are the same size)
                    w, h = Image.open(filename_filter_rgb_list[0]).size

                    # create a numpy array of floats to store the average (assume RGB images)
                    arr = np.zeros((h, w, 4), np.float)
                    # build up average pixel intensities, casting each image as an array of floats
                    for im in filename_filter_rgb_list:
                        im_arr = np.array(Image.open(im), dtype=np.float)
                        arr = arr + im_arr

                    # round values in array and cast as 8-bit integer
                    arr = np.array(np.round(arr), dtype=np.uint8)
                    # generate, save and preview final image
                    img = Image.fromarray(arr, mode="RGBA")

                    # save filter sum
                    filter_rgb_name = str(target_layer) + "_" + "filter_nb" + str(idx_filter) + \
                        "_image_" + str(idx_img) + "_all_ch" + ".png"
                    filename_filter_rgb = os.path.abspath(os.path.normpath(os.path.join(layer_dir, filter_rgb_name)))
                    img.save(filename_filter_rgb)

        if plot_filter_sum:
            filename_filter_sum_rgb_layers = []
            for idx_layer, target_layer in enumerate(target_layers):
                filename_filter_sum_rgb_list = []

                # make a dir for each target layer
                layer_dir = os.path.abspath(os.path.normpath(os.path.join(figure_dir, target_layer)))
                if not os.path.exists(layer_dir):
                    os.makedirs(layer_dir)

                for idx_ch in range(nb_channels):
                    plt.clf()

                    # data_filter = output_for_filters[idx_layer, :, :, :, :]
                    data_filter_ch = output_for_filters[idx_layer, :, idx_ch, :, :]

                    # sum up contributions from all filters to have the whole range of
                    # possibilities
                    # data_filter_ch is #filters, px, py
                    combined_filters = data_filter_ch.sum(axis=0)
                    combined_filters = combined_filters.reshape(img_size[0], img_size[1])
                    plt.axis('off')

                    # plt.imshow(input_picture, cmap='gray', vmin=0, vmax=0.05)
                    plt.imshow(combined_filters, alpha=0.9, cmap=cmaps[idx_ch])
                    # show only positive filter
                    vmin = max(0.0, amin(combined_filters))
                    vmax = amax(combined_filters)

                    plt.imshow(combined_filters, cmap=cmaps[idx_ch], vmin=vmin, vmax=vmax * 1.0)

                    # save filter sum
                    filter_sum_name = str(target_layer) + "_image_" + str(idx_img) + "_ch" + str(idx_ch) + "_sum.png"
                    filename_filter_sum = os.path.abspath(os.path.normpath(os.path.join(layer_dir, filter_sum_name)))
                    # to avoid whitespaces when saving
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(filename_filter_sum, dpi=100, bbox_inches='tight', pad_inches=0.0, format='png')

                    # append to calculate the overlayed rgb picture
                    filename_filter_sum_rgb_list.append(filename_filter_sum)

                # plot the rgb sum of all filters
                # Assuming all images are the same size, get dimensions of first image
                w, h = Image.open(filename_filter_sum_rgb_list[0]).size

                # Create a numpy array of floats to store the average (assume RGB images)
                arr = np.zeros((h, w, 4), np.float)
                # Build up average pixel intensities, casting each image as an array of floats
                for im in filename_filter_sum_rgb_list:
                    im_arr = np.array(Image.open(im), dtype=np.float)
                    arr = arr + im_arr

                # Round values in array and cast as 8-bit integer
                arr = np.array(np.round(arr), dtype=np.uint8)
                # Generate, save and preview final image
                img = Image.fromarray(arr, mode="RGBA")

                # save filter sum
                filter_sum_rgb_name = "all_ch_" + str(target_layer) + "_image_" + str(idx_img) + "_sum.png"
                filename_filter_sum_rgb = os.path.abspath(
                    os.path.normpath(os.path.join(layer_dir, filter_sum_rgb_name)))
                img.save(filename_filter_sum_rgb)

            # list of the rgb sum of filters for all layers
            filename_filter_sum_rgb_layers.append(filename_filter_sum_rgb)

        if plot_summary:
            if plot_filter_sum:
                # for each channel plot summary
                for idx_ch in range(nb_channels):
                    filename_summary = str(idx_img) + "_summary_plot_ch" + str(idx_ch) + ".png"
                    filename_summary_full = os.path.abspath(
                        os.path.normpath(os.path.join(figure_dir, filename_summary)))

                    plt.clf()
                    fig = plt.figure(figsize=(8, 6))
                    plt.style.use('fivethirtyeight')
                    input_img = plt.imread(filename_input_full)
                    att_resp_maps = plt.imread(filenames_ch[idx_ch])
                    # plot sum of filters from last layer
                    last_layer_filter_sum = plt.imread(filename_filter_sum_rgb)

                    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])
                    gs.update(left=0.05, right=0.90, wspace=0.05, hspace=0.0)

                    ax0 = plt.subplot(gs[0])
                    ax0.imshow(input_img)
                    ax0.axis('off')

                    ax1 = plt.subplot(gs[1])
                    ax1.imshow(att_resp_maps)
                    ax1.axis('off')

                    ax2 = plt.subplot(gs[2])
                    ax2.imshow(last_layer_filter_sum)
                    ax2.axis('off')

                    plt.savefig(filename_summary_full, dpi=100, bbox_inches='tight', format='png')
                    plt.close()
            else:
                raise Exception("Cannot produce plot summary without filter sum. Set plot_filter_sum=True.")


def deconv_visualize(model, target_layer, input_data, nb_top_feat_maps):
    """Obtain attentive response maps back-projected to image space using transposed convolutions
    (sometimes referred as deconvolutions in machine learning).

    Parameters:

    model: instance of the Keras model
        The ConvNet model to be used.

    target_layer: str
        Name of the layer for which we want to obtain the attentive response maps. The names of the layers are defined
        in the Keras instance ``model``.

    input_data: ndarray
        The image data to be passed through the network. Shape: (n_samples, n_channels, img_dim1, img_dim2)

    nb_top_feat_maps: int
        Top-n filter you want to visualize, e.g. nb_top_feat_maps = 25 will visualize top 25 filters in target layer

    .. codeauthor:: Devinder Kumar <d22kumar@uwaterloo.ca>

    """

    feat_maps = None

    Dec = DeconvNet(model)

    deconv_imgs_batch = np.random.random(
        (input_data.shape[0],
         nb_top_feat_maps,
         input_data.shape[1],
         input_data.shape[2],
         input_data.shape[3]))
    logger.info('Size of deconv batch init: {}'.format(deconv_imgs_batch.shape))
    logger.info('Using top {} filters'.format(nb_top_feat_maps))

    for img_index in range(input_data.shape[0]):
        feat_maps = get_max_activated_filter_for_layer(target_layer, model, input_data, nb_top_feat_maps, img_index)
        logger.info('input data shape : {}'.format(input_data.shape))
        deconv_imgs = get_deconv_imgs(img_index, input_data, Dec, target_layer, feat_maps)
        deconv_imgs_batch[img_index, :, :, :] = deconv_imgs

    return deconv_imgs_batch, feat_maps


def get_max_activated_filter_for_layer(target_layer, model, input_data, nb_top_feat_maps, img_index):
    """Find the indices of the most activated filters for a given image in the specified target layer of a Keras model.

    Parameters:

    target_layer: str
        Name of the layer for which we want to obtain the attentive response maps. The names of the layers are defined in the Keras instance ``model``.

    model: instance of the Keras model
        The ConvNet model to be used.

    input_data:
        input_data: ndarray
        The image data to be passed through the network. Shape: (n_samples, n_channels, img_dim1, img_dim2)

    nb_top_feat_maps:
        Number of the top attentive response maps to be calculated and plotted.
        It must be <= to the minimum number of filters used in the  neural network layers. This is not checked by the code, and
        respecting this criterion is up to the user.

    img_index: list or ndarray
        Array or list of index. These are the indices of the images (contained in ``data``) for which we want to obtain the attentive response maps.

    Returns: list of int
        List containing the indices of the filters with the highest response (activation) for the given image.

    .. codeauthor:: Devinder Kumar <d22kumar@uwaterloo.ca>
    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    img = input_data[img_index]
    if len(img.shape) < 3:
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
    elif len(img.shape) < 4:
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # get the output of the target layer
    # see for example https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    layer_output = layer_dict[target_layer].output

    # keras.backend.function(inputs, outputs)
    # the Keras backend function is first created with placeholders and then the actual valued are passed
    response = K.function([model.input, K.learning_phase()], [layer_output])
    output_response = response([img, 0])
    filtered_images = ((output_response[0])[0])

    # find the filtered images with the highest activation
    sort_filter = []
    for im in filtered_images:
        im_sum = np.sum(im)
        sort_filter.append(im_sum)

    index_sort = np.argsort(sort_filter)
    # reversing the sorted list
    index_sort = index_sort[::-1]
    top_n_filter = index_sort[:nb_top_feat_maps]

    # top_filtered_images = np.asarray([filtered_images.tolist()[i] for i in top_n_filter])

    return top_n_filter


def get_deconv_imgs(img_index, data, dec_layer, target_layer, feat_maps):
    """Return the attentive response maps of the images specified in img_index for the target layer and feature maps
    specified in the arguments.

    Parameters:

    img_index: list or ndarray
        Array or list of index. These are the indices of the images (contained in ``data``) for which we want to obtain the attentive response maps.

    data: ndarray
        The image data. Shape : (n_samples, n_channels, img_dim1, img_dim2)

    Dec: instance of class :py:mod:`ai4materials.interpretation.deconv_resp_maps.DeconvNet`
        DeconvNet model: instance of the DeconvNet class

    target_layer: str
        Name of the layer for which we want to obtain the attentive response maps. The names of the layers are defined in the Keras instance ``model``.

    feat_map: int
        Index of the attentive response map to visualise.

    .. codeauthor:: Devinder Kumar <d22kumar@uwaterloo.ca>

    """

    X_deconv = []
    X_orig = data[img_index]

    logger.debug('Shape of original images: {}'.format(X_orig.shape))
    if len(X_orig.shape) < 3:
        X_orig = X_orig.reshape(1, 1, X_orig.shape[0], X_orig.shape[1])
    elif len(X_orig.shape) < 4:
        X_orig = X_orig.reshape(1, X_orig.shape[0], X_orig.shape[1], X_orig.shape[2])

    for feat_map in feat_maps:
        X_deconv_img = dec_layer.get_deconv(X_orig, target_layer, feat_map=feat_map)
        # from keras.layers import Deconv2D
        # X_deconv_img = Deconv2D(x, kernel, output_shape, strides=(1, 1), border_mode='valid',
        #          dim_ordering='default', image_shape=None, filter_shape=None)

        logger.info('Shape of images from transposed convolution: {}'.format(X_deconv_img.shape))
        X_deconv.append(X_deconv_img[0])
        logger.debug(X_deconv_img.shape)

    X_deconv = np.asarray(X_deconv)

    return X_deconv


class DeconvNet(object):
    """DeconvNet class. Code taken from:
    https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DeconvNet/KerasDeconv.py

    """

    def __init__(self, model):
        self.model = model
        list_layers = self.model.layers
        self.lnames = [l.name for l in list_layers]
        assert len(self.lnames) == len(set(self.lnames)), "Non unique layer names"
        # Dict of layers indexed by layer name
        self.d_layers = {}
        for l_name, l in zip(self.lnames, list_layers):
            self.d_layers[l_name] = l

        # Tensor for function definitions
        self.x = K.T.tensor4('x')

    def __getitem__(self, layer_name):
        try:
            return self.d_layers[layer_name]
        except KeyError:
            logger.debug("Erroneous layer name")

    def _deconv(self, X, lname, d_switch, feat_map=None):
        o_width, o_height = self[lname].output_shape[-2:]

        # Get filter size
        f_width = self[lname].W_shape[2]
        f_height = self[lname].W_shape[3]

        # Keras 2.0.
        # f_width = self[lname].kernel_size[0]
        # f_height = self[lname].kernel_size[1]

        # Compute padding needed
        i_width, i_height = X.shape[-2:]
        pad_width = (o_width - i_width + f_width - 1) / 2
        pad_height = (o_height - i_height + f_height - 1) / 2

        pad_width = int(pad_width)
        pad_height = int(pad_height)

        assert isinstance(pad_width, int), "Pad width size issue at layer %s" % lname
        assert isinstance(pad_height, int), "Pad height size issue at layer %s" % lname

        # Set to zero based on switch values
        X[d_switch[lname]] = 0
        # Get activation function
        activation = self[lname].activation
        X = activation(X)
        if feat_map is not None:
            logger.debug("Setting other feat map to zero")
            for i in range(X.shape[1]):
                if i != feat_map:
                    X[:, i, :, :] = 0
            logger.debug("Setting non max activations to zero")
            for i in range(X.shape[0]):
                iw, ih = np.unravel_index(X[i, feat_map, :, :].argmax(), X[i, feat_map, :, :].shape)
                m = np.max(X[i, feat_map, :, :])
                X[i, feat_map, :, :] = 0
                X[i, feat_map, iw, ih] = m
        # Get filters. No bias for now
        W = self[lname].W
        # W = self[lname].kernel    # keras 2.0

        # Transpose filter
        W = W.transpose([1, 0, 2, 3])
        W = W[:, :, ::-1, ::-1]
        # CUDNN for conv2d ?
        conv_out = K.T.nnet.conv2d(input=self.x, filters=W, border_mode='valid')
        # Add padding to get correct size
        pad = K.function([self.x], K.spatial_2d_padding(self.x, padding=(pad_width, pad_height), dim_ordering="th"))

        # Keras 2.0 but not sure
        # pad = K.function([self.x], K.spatial_2d_padding(
        #     self.x, padding=((pad_width, 0), (0, pad_height)), data_format="channels_first"))

        X_pad = pad([X])
        # Get Deconv output
        deconv_func = K.function([self.x], conv_out)
        X_deconv = deconv_func([X_pad])
        assert X_deconv.shape[-2:] == (o_width, o_height), "Deconv output at %s has wrong size" % lname
        return X_deconv

    def _forward_pass(self, X, target_layer):

        # For all layers up to the target layer
        # Store the max activation in switch
        d_switch = {}
        layer_index = self.lnames.index(target_layer)
        for lname in self.lnames[:layer_index + 1]:
            # Get layer output
            inc, out = self[lname].input, self[lname].output
            f = K.function([inc, K.learning_phase()], out)
            X = f([X, 0])
            if "convolution2d" in lname:
                d_switch[lname] = np.where(X <= 0)
        return d_switch

    def _backward_pass(self, X, target_layer, d_switch, feat_map):
        # Run deconv/maxunpooling until input pixel space
        layer_index = self.lnames.index(target_layer)
        # Get the output of the target_layer of interest
        layer_output = K.function([self[self.lnames[0]].input, K.learning_phase()], self[target_layer].output)
        X_outl = layer_output([X, 0])
        # Special case for the starting layer where we may want
        # to switchoff somes maps/ activations
        logger.debug("Deconvolving %s..." % target_layer)
        if "maxpooling2d" in target_layer:
            X_maxunp = K.pool.max_pool_2d_same_size(self[target_layer].input, self[target_layer].pool_size)
            unpool_func = K.function([self[self.lnames[0]].input], X_maxunp)
            X_outl = unpool_func([X])
            if feat_map is not None:
                for i in range(X_outl.shape[1]):
                    if i != feat_map:
                        X_outl[:, i, :, :] = 0
                for i in range(X_outl.shape[0]):
                    iw, ih = np.unravel_index(X_outl[i, feat_map, :, :].argmax(), X_outl[i, feat_map, :, :].shape)
                    m = np.max(X_outl[i, feat_map, :, :])
                    X_outl[i, feat_map, :, :] = 0
                    X_outl[i, feat_map, iw, ih] = m
        elif "convolution2d" in target_layer:
            X_outl = self._deconv(X_outl, target_layer, d_switch, feat_map=feat_map)
        else:
            raise ValueError("Invalid layer name: %s \n Can only handle maxpool and conv" % target_layer)
        # Iterate over layers (deepest to shallowest)
        for lname in self.lnames[:layer_index][::-1]:
            logger.debug("Deconvolving %s..." % lname)
            # Unpool, Deconv or do nothing
            if "maxpooling2d" in lname:
                p1, p2 = self[lname].pool_size
                uppool = K.function([self.x], K.resize_images(self.x, p1, p2, "th"))
                X_outl = uppool([X_outl])

            elif "convolution2d" in lname:
                X_outl = self._deconv(X_outl, lname, d_switch)
            elif "padding" in lname:
                pass
            elif "batchnormalization" in lname:
                pass
            elif "input" in lname:
                pass
            else:
                raise ValueError(
                    "Invalid layer name: %s \n Can only handle maxpool, conv, bacthnormalization and padding layer" % lname)
        return X_outl

    def get_layers(self):
        list_layers = self.model.layers
        list_layers_name = [l.name for l in list_layers]
        return list_layers_name

    def get_deconv(self, X, target_layer, feat_map=None):

        # First make predictions to get feature maps
        self.model.predict(X)
        # Forward pass storing switches
        logger.debug("Starting forward pass...")
        start_time = time.time()
        d_switch = self._forward_pass(X, target_layer)
        end_time = time.time()
        logger.debug('Forward pass completed in %ds' % (end_time - start_time))
        # Then deconvolve starting from target layer
        X_out = self._backward_pass(X, target_layer, d_switch, feat_map)
        return X_out
