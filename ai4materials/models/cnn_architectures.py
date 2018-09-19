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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Convolution3D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling3D
from keras.layers import GlobalAveragePooling2D
import logging

logger = logging.getLogger('ai4materials')


def model_fully_conv(conv2d_filters, kernel_sizes, n_rows, n_columns, img_channels, nb_classes):
    model = Sequential()
    model.add(Conv2D(conv2d_filters[0], kernel_sizes[0], kernel_sizes[0], name='convolution2d_1', activation='relu',
                     border_mode='same', init='orthogonal', bias=True,
                     input_shape=(img_channels, n_rows, n_columns)))
    # model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    # model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(conv2d_filters[1], kernel_sizes[1], kernel_sizes[1], name='convolution2d_2', activation='relu',
                     border_mode='same', init='orthogonal', bias=True))
    # model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    # model.add(Dropout(0.25, name='dropout_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_1'))

    model.add(Conv2D(conv2d_filters[2], kernel_sizes[2], kernel_sizes[2], name='convolution2d_3', activation='relu',
                     border_mode='same', init='orthogonal', bias=True))
    # model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    # model.add(Dropout(0.25, name='dropout_3'))

    model.add(Conv2D(conv2d_filters[3], kernel_sizes[3], kernel_sizes[3], name='convolution2d_4', activation='relu',
                     border_mode='same', init='orthogonal', bias=True))
    # model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    # model.add(Dropout(0.25, name='dropout_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_2'))

    model.add(Conv2D(conv2d_filters[4], kernel_sizes[4], kernel_sizes[4], name='convolution2d_5', activation='relu',
                     border_mode='same', init='orthogonal', bias=True))
    # model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))

    model.add(Conv2D(conv2d_filters[5], kernel_sizes[4], kernel_sizes[4], name='convolution2d_6', activation='relu',
                     border_mode='same', init='orthogonal', bias=True))

    # 1x1 convolution
    model.add(Conv2D(conv2d_filters[6], 1, 1, name='convolution2d_7', activation='relu', border_mode='same',
                     init='orthogonal', bias=True))

    model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.10, name='dropout_1'))

    model.add(GlobalAveragePooling2D(dim_ordering='default'))
    model.add(Dense(nb_classes, name='dense_1'))
    model.add(Activation('softmax', name='activation_1'))

    return model


def model_cnn_rot_inv(conv2d_filters, kernel_sizes, hidden_layer_size, n_rows, n_columns,
                      img_channels, nb_classes, dropout):
    """Deep convolutional neural network model for crystal structure recognition.

    This neural network architecture was used to classify crystal structures - represented by the three-dimensional
    diffraction fingerprint - in Ref. [1]_.


    .. [1] A. Ziletti et al.,
        “Automatic structure identification in polycrystals via Bayesian deep learning”,
        in preparation (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    n_conv2d = 6

    if not len(conv2d_filters) == n_conv2d:
        raise Exception(
            "Wrong number of filters. Give a list of {0} numbers.".format(n_conv2d))
    if not len(kernel_sizes) == n_conv2d:
        raise Exception(
            "Wrong number of kernel sizes. Give a list of {0} numbers.".format(n_conv2d))

    model = Sequential()
    model.add(
        Conv2D(conv2d_filters[0], kernel_sizes[0], kernel_sizes[0], name='convolution2d_1', activation='relu',
               border_mode='same', init='orthogonal', bias=True, input_shape=(img_channels, n_rows, n_columns)))
    model.add(
        Conv2D(conv2d_filters[1], kernel_sizes[1], kernel_sizes[1], name='convolution2d_2', activation='relu',
               border_mode='same', init='orthogonal', bias=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_1'))
    model.add(
        Conv2D(conv2d_filters[2], kernel_sizes[2], kernel_sizes[2], name='convolution2d_3', activation='relu',
               border_mode='same', init='orthogonal', bias=True))
    model.add(
        Conv2D(conv2d_filters[3], kernel_sizes[3], kernel_sizes[3], name='convolution2d_4', activation='relu',
               border_mode='same', init='orthogonal', bias=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_2'))
    model.add(
        Conv2D(conv2d_filters[4], kernel_sizes[4], kernel_sizes[4], name='convolution2d_5', activation='relu',
               border_mode='same', init='orthogonal', bias=True))
    model.add(
        Conv2D(conv2d_filters[5], kernel_sizes[5], kernel_sizes[5], name='convolution2d_6', activation='relu',
               border_mode='same', init='orthogonal', bias=True))

    model.add(Flatten(name='flatten_1'))
    # model.add(BatchNormalization())
    model.add(Dense(hidden_layer_size, name='dense_1', activation='relu', bias=True))
    model.add(Dropout(dropout, name='dropout_1'))
    model.add(Dense(nb_classes, name='dense_2'))
    model.add(Activation('softmax', name='activation_1'))

    return model


def cnn_nature_comm_ziletti2018(conv2d_filters, kernel_sizes, max_pool_strides, hidden_layer_size, n_rows,
                                n_columns, img_channels, nb_classes):
    """Deep convolutional neural network model for crystal structure recognition.

    This neural network architecture was used to classify crystal structures - represented by the two-dimensional
    diffraction fingerprint - in Ref. [2]_


    .. [2] A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli,
        “Insightful classification of crystal structures using deep learning”,
        Nature Communications, vol. 9, pp. 2775 (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    n_conv_2d = 6
    n_pool = 2
    if not len(conv2d_filters) == n_conv_2d:
        raise Exception(
            "Wrong number of filters. Give a list of {0} numbers.".format(n_conv_2d))
    if not len(kernel_sizes) == n_conv_2d:
        raise Exception(
            "Wrong number of kernel sizes. Give a list of {0} numbers.".format(n_conv_2d))
    if not len(max_pool_strides) == n_pool:
        raise Exception(
            "Wrong number of max pool strides. Give a list of {0} numbers.".format(n_pool))

    model = Sequential()
    model.add(
        Convolution2D(conv2d_filters[0], kernel_sizes[0], kernel_sizes[0], name='convolution2d_1', activation='relu',
                      border_mode='same', init='orthogonal', bias=True, input_shape=(img_channels, n_rows, n_columns)))
    model.add(
        Convolution2D(conv2d_filters[1], kernel_sizes[1], kernel_sizes[1], name='convolution2d_2', activation='relu',
                      border_mode='same', init='orthogonal', bias=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_1'))
    model.add(
        Convolution2D(conv2d_filters[2], kernel_sizes[2], kernel_sizes[2], name='convolution2d_3', activation='relu',
                      border_mode='same', init='orthogonal', bias=True))
    model.add(
        Convolution2D(conv2d_filters[3], kernel_sizes[3], kernel_sizes[3], name='convolution2d_4', activation='relu',
                      border_mode='same', init='orthogonal', bias=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_2'))
    model.add(
        Convolution2D(conv2d_filters[4], kernel_sizes[4], kernel_sizes[4], name='convolution2d_5', activation='relu',
                      border_mode='same', init='orthogonal', bias=True))
    model.add(
        Convolution2D(conv2d_filters[5], kernel_sizes[5], kernel_sizes[5], name='convolution2d_6', activation='relu',
                      border_mode='same', init='orthogonal', bias=True))

    model.add(Flatten(name='flatten_1'))
    model.add(Dropout(0.25, name='dropout_1'))
    model.add(Dense(hidden_layer_size, name='dense_1', activation='relu', bias=True))
    model.add(BatchNormalization())

    model.add(Dense(nb_classes, name='dense_2'))
    model.add(Activation('softmax', name='activation_1'))

    return model


def model_architecture_3d(dim1, dim2, dim3, img_channels, nb_classes):
    model = Sequential()
    model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', init='orthogonal', bias=True,
                            input_shape=(img_channels, dim1, dim2, dim3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same', init='orthogonal', bias=True))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Convolution3D(8, 3, 3, 3, activation='relu', border_mode='same', init='orthogonal', bias=True))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', bias=True))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model
