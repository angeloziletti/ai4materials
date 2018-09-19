#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016-2018, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "03/05/18"

from functools import partial
import keras
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.models import model_from_json
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
import logging
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import six.moves.cPickle as pickle
logger = logging.getLogger('nomad-ml')
logging.basicConfig(level='DEBUG')


def load_data(path_to_x, path_to_y):
    logger.debug("Loading X from {}".format(path_to_x))
    logger.debug("Loading y from {}".format(path_to_y))

    with open(path_to_x, 'rb') as input_x:
        x = pickle.load(input_x).astype('float32')

    with open(path_to_y, 'rb') as input_y:
        y = pickle.load(input_y).astype('float32')

    logger.debug('X-shape: {0}'.format(x.shape))
    logger.debug('y-shape: {0}'.format(y.shape))

    return x, y


def model_deep_cnn_struct_recognition(conv2d_filters, kernel_sizes, max_pool_strides, hidden_layer_size, n_rows,
                                      n_columns, img_channels, nb_classes):
    """Deep convolutional neural network model for crystal structure recognition.

    Examples
    --------
    Suggested parameters::

        partial_model_architecture = partial(
        model_deep_cnn_struct_recognition,
        conv2d_filters=[32, 16, 12, 12, 8, 8],
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        max_pool_strides=[2, 2])

    """

    N_CONV_2D = 6
    N_POOL = 2
    if not len(conv2d_filters) == N_CONV_2D: raise Exception(
        "Wrong number of filters. Give a list of {0} numbers.".format(N_CONV_2D))
    if not len(kernel_sizes) == N_CONV_2D: raise Exception(
        "Wrong number of kernel sizes. Give a list of {0} numbers.".format(N_CONV_2D))
    if not len(max_pool_strides) == N_POOL: raise Exception(
        "Wrong number of max pool strides. Give a list of {0} numbers.".format(N_POOL))

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

    model.add(Dropout(0.25, name='dropout_1'))
    model.add(Flatten(name='flatten_1'))
    model.add(Dense(hidden_layer_size, name='dense_1', activation='relu', bias=True))
    model.add(BatchNormalization())

    model.add(Dense(nb_classes, name='dense_2'))
    model.add(Activation('softmax', name='activation_1'))

    return model


def train_cnn_keras(x_train, y_train, checkpoint_filename, batch_size, nb_classes, nb_epoch, img_channels,
                    partial_model_architecture, normalize=False):

    # the 1st dimension is the batch (nb of images)
    input_dims = (x_train.shape[1], x_train.shape[2])

    if len(input_dims) == 2:
        # add channels
        if keras.backend.image_dim_ordering() == 'th':
            x_train = np.reshape(x_train, (x_train.shape[0], -1, x_train.shape[1], x_train.shape[2]))
        elif keras.backend.image_dim_ordering() == 'tf':
            raise NotImplementedError('Tensorflow backend is not supported.')
        else:
            raise ValueError('Image ordering type not recognized. Possible values are th or tf.')
    else:
        raise Exception("Wrong number of dimensions.")

    logger.info('Loading datasets.')

    logger.info('x_train shape: {0}'.format(x_train.shape))
    logger.info('y_train shape: {0}'.format(y_train.shape))
    logger.info('Training samples: {0}'.format(x_train.shape[0]))

    # normalize each image separately
    if normalize:
        for idx in range(x_train.shape[0]):
            x_train[idx, :, :, :] = (x_train[idx, :, :, :] - np.amin(x_train[idx, :, :, :])) / (
                    np.amax(x_train[idx, :, :, :]) - np.amin(x_train[idx, :, :, :]))

    # check if the image is already normalized
    logger.info(
        'Maximum value in x_train for the 1st image (to check normalization): {0}'.format(np.amax(x_train[0, :, :, :])))

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)

    logger.info('Loading and formatting of data completed.')

    # return the Keras model
    model = partial_model_architecture(n_rows=input_dims[0], n_columns=input_dims[1], img_channels=img_channels,
                                       nb_classes=nb_classes)

    # serialize model to JSON
    model_json = model.to_json()
    with open(checkpoint_filename + ".json", "w") as json_file:
        json_file.write(model_json)

    callbacks = []
    save_model_per_epoch = ModelCheckpoint(checkpoint_filename + ".h5", monitor='val_acc', verbose=1,
                                           save_best_only=False, mode='max', period=1)

    callbacks.append(save_model_per_epoch)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1, shuffle=True, verbose=1,
              callbacks=callbacks)

    model.save(checkpoint_filename + ".h5")
    logger.info("Model saved to disk.")
    logger.debug("Filename: {0}".format(checkpoint_filename))
    del model


def predict_cnn_keras(x_test, y_test, nb_classes,
                      checkpoint_filename, batch_size=32,
                      normalize=False,
                      verbose=1):

    # loading saved model
    model_arch_file = checkpoint_filename + ".json"
    model_weights_file = checkpoint_filename + ".h5"

    json_file = open(model_arch_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    logger.debug('Loading model weights.')
    loaded_model.load_weights(model_weights_file)
    logger.info('Model loaded correctly.')

    # evaluate loaded model on test data
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    loaded_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # convert class vectors to binary class matrices
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # Theano backend uses (nb_sample, channels, height, width)
    input_dims = (x_test.shape[1], x_test.shape[2])

    # works only for Keras 1.
    if len(input_dims) == 2:
        # add channels
        if keras.backend.image_dim_ordering() == 'th':
            x_test = np.reshape(x_test, (x_test.shape[0], -1, x_test.shape[1], x_test.shape[2]))
        elif keras.backend.image_dim_ordering() == 'tf':
            raise NotImplementedError('Tensorflow backend is not supported.')
        else:
            raise ValueError('Image ordering type not recognized. Possible values are th or tf.')
    else:
        raise Exception("Wrong number of dimensions.")

    logger.info('Loading test dataset for prediction.')

    logger.debug('x_test shape: {0}'.format(x_test.shape))
    logger.debug('Test samples: {0}'.format(x_test.shape[0]))

    # normalize each image separately
    if normalize:
        if len(input_dims) == 2:
            for idx in range(x_test.shape[0]):
                x_test[idx, :, :, :] = (x_test[idx, :, :, :] - np.amin(x_test[idx, :, :, :])) / (
                        np.amax(x_test[idx, :, :, :]) - np.amin(x_test[idx, :, :, :]))
        else:
            for idx in range(x_test.shape[0]):
                x_test[idx, :, :, :] = (x_test[idx, :, :, :, :] - np.amin(x_test[idx, :, :, :, :])) / (
                        np.amax(x_test[idx, :, :, :, :]) - np.amin(x_test[idx, :, :, :, :]))

    # check if the image is already normalized
    if len(input_dims) == 2:
        logger.debug('Maximum value in x_test for the 1st image (to check normalization): {0}'.format(
            np.amax(x_test[0, :, :, :])))
    else:
        logger.debug('Maximum value in x_test for the 1st image (to check normalization): {0}'.format(
            np.amax(x_test[0, :, :, :, :])))

    # convert class vectors to binary class matrices
    # y_test = np_utils.to_categorical(y_test, nb_classes)
    logger.info('Loading and formatting of data completed.')
    loaded_model.summary()

    logger.info('Predicting...')

    # compiling and calculating the score of the model again
    score = loaded_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
    logger.info('Model score: {0} {1}%'.format(loaded_model.metrics_names[1], score[1] * 100))

    y_pred = loaded_model.predict_classes(x_test, batch_size=batch_size, verbose=verbose)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    logger.info('Confusion matrix, without normalization: ')
    logger.info(conf_matrix)


if __name__ == "__main__":
    # the scripts was check with the following dependencies:
    # Keras == 1.2.0
    # tensorflow==0.9.0
    # scikit-learn==0.19.1
    # Theano 0.9.0

    # folder where the datasets are saved
    dataset_train_dir = '/home/ziletti/Documents/calc_xray/2d_nature_comm/datasets_2d/training/'
    dataset_test_dir = '/home/ziletti/Documents/calc_xray/2d_nature_comm/datasets_2d/test/'

    # folder where the Keras models are saved
    checkpoint_dir = '/home/ziletti/Documents/calc_xray/2d_nature_comm/saved_models'
    # path to the file where the Keras model is saved or loaded from
    checkpoint_filename = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'trial_nn')))

    # filename with the pre-trained network
    # checkpoint_filename = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'ziletti_et_2018_rgb')))

    # training dataset
    path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(dataset_train_dir, 'pristine_dataset_x.pkl')))
    path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(dataset_train_dir, 'pristine_dataset_y.pkl')))
    x_train, y_train = load_data(path_to_x_train, path_to_y_train)

    # one of the possible test sets
    path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_test_dir, 'vac0.01_dataset_x.pkl')))
    path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_test_dir, 'vac0.01_dataset_y.pkl')))
    x_test, y_test = load_data(path_to_x_test, path_to_y_test)

    # feed some user-defined number in the template Keras model
    partial_model_architecture = partial(model_deep_cnn_struct_recognition, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                         kernel_sizes=[7, 7, 7, 7, 7, 7], max_pool_strides=[2, 2],
                                         hidden_layer_size=128)

    # nb of different classes in the dataset
    nb_classes = len(list(set(y_train)))

    train_cnn_keras(x_train, y_train, batch_size=32, nb_classes=nb_classes, nb_epoch=1, img_channels=3,
                    partial_model_architecture=partial_model_architecture,
                    normalize=False, checkpoint_filename=checkpoint_filename)

    predict_cnn_keras(x_test, y_test, nb_classes=nb_classes,
                      checkpoint_filename=checkpoint_filename, batch_size=32,
                      normalize=False,
                      verbose=1)
