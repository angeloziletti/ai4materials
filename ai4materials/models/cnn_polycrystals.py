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

import keras
import keras.backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ai4materials.utils.utils_plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from scipy import stats
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
import sys
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger('ai4materials')


def train_neural_network(data_set, configs, batch_size, nb_classes, nb_epoch, img_channels,
                    partial_model_architecture, normalize=False, checkpoint_dir=None, checkpoint_filename=None,
                    training_log_file='training.log', early_stopping=False):
    filename_no_ext = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename)))

    training_log_file_path = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, training_log_file)))

    x_train = data_set.train.images
    y_train = data_set.train.labels
    x_val = data_set.val.images
    y_val = data_set.val.labels
    x_test = data_set.test.images
    y_test = data_set.test.labels

    # image conventions
    # - TensorFlow: [batch, width, height, channels]
    # - Theano: [batch, channels, width, height]

    # the 1st dimension is the batch (nb of images)
    input_dims = (x_train.shape[1], x_train.shape[2])

    if len(input_dims) == 2:
        # add channels
        if keras.backend.image_dim_ordering() == 'th':
            x_train = np.reshape(x_train, (x_train.shape[0], -1, x_train.shape[1], x_train.shape[2]))
            x_val = np.reshape(x_val, (x_val.shape[0], -1, x_val.shape[1], x_val.shape[2]))
            x_test = np.reshape(x_test, (x_test.shape[0], -1, x_test.shape[1], x_test.shape[2]))
        elif keras.backend.image_dim_ordering() == 'tf':
            raise NotImplementedError('Tensorflow backend is not supported.')
        else:
            raise ValueError('Image ordering type not recognized. Possible values are th or tf.')
    else:
        raise Exception("Wrong number of dimensions.")

    logger.info('Loading datasets.')

    logger.debug('x_train shape: {0}'.format(x_train.shape))
    logger.debug('y_train shape: {0}'.format(y_train.shape))
    logger.debug('x_val shape: {0}'.format(x_val.shape))
    logger.debug('y_val shape: {0}'.format(y_val.shape))
    logger.debug('x_test shape: {0}'.format(x_test.shape))
    logger.debug('y_test shape: {0}'.format(y_test.shape))
    logger.debug('Training samples: {0}'.format(x_train.shape[0]))
    logger.debug('Validation samples: {0}'.format(x_val.shape[0]))
    logger.debug('Test samples: {0}'.format(x_test.shape[0]))

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')

    # normalize each image separately
    if normalize:
        for idx in range(x_train.shape[0]):
            x_train[idx, :, :, :] = (x_train[idx, :, :, :] - np.amin(x_train[idx, :, :, :])) / (
                np.amax(x_train[idx, :, :, :]) - np.amin(x_train[idx, :, :, :]))
        for idx in range(x_val.shape[0]):
            x_val[idx, :, :, :] = (x_val[idx, :, :, :] - np.amin(x_val[idx, :, :, :])) / (
                np.amax(x_val[idx, :, :, :]) - np.amin(x_val[idx, :, :, :]))
        for idx in range(x_test.shape[0]):
            x_test[idx, :, :, :] = (x_test[idx, :, :, :] - np.amin(x_test[idx, :, :, :])) / (
                np.amax(x_test[idx, :, :, :]) - np.amin(x_test[idx, :, :, :]))

    # check if the image is already normalized
    logger.info(
        'Maximum value in x_train for the 1st image (to check normalization): {0}'.format(np.amax(x_train[0, :, :, :])))
    logger.info(
        'Maximum value in x_val for the 1st image (to check normalization): {0}'.format(np.amax(x_val[0, :, :, :])))
    logger.info(
        'Maximum value in x_test for the 1st image (to check normalization): {0}'.format(np.amax(x_test[0, :, :, :])))

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    logger.info('Loading and formatting of data completed.')

    # return the Keras model
    model = partial_model_architecture(n_rows=input_dims[0], n_columns=input_dims[1], img_channels=img_channels,
                                       nb_classes=nb_classes)

    logger.info("{}".format(model.summary()))
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename_no_ext + ".json", "w") as json_file:
        json_file.write(model_json)

    callbacks = []
    csv_logger = CSVLogger(training_log_file_path, separator=',', append=False)
    save_model_per_epoch = ModelCheckpoint(filename_no_ext + ".h5", monitor='val_acc', verbose=1, save_best_only=True,
                                           mode='max', period=1)
    callbacks.append(csv_logger)
    callbacks.append(save_model_per_epoch)

    # if you are running on Notebook
    if configs['runtime']['isBeaker']:
        callbacks.append(TQDMNotebookCallback(leave_inner=True, leave_outer=True))
    else:
        callbacks.append(TQDMCallback(leave_inner=True, leave_outer=True))

    if early_stopping:
        EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=0, mode='auto')

    # adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    logger.info("Using test set instead of validation set")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(x_test, y_test), shuffle=True, verbose=0, callbacks=callbacks)

    # serialize weights to HDF5
    # model.save(filename_no_ext + ".h5")
    logger.info("Model saved to disk.")
    logger.debug("Filename: {0}".format(filename_no_ext))
    del model


def predict(data_set, nb_classes, configs, results_file, numerical_labels, text_labels, checkpoint_dir=None,
                      checkpoint_filename=None, batch_size=1, show_model_acc=True, mc_samples=1,
                      predict_probabilities=True, plot_conf_matrix=True, conf_matrix_file=None, normalize=False,
                      verbose=1):
    filename_no_ext = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename)))

    if verbose is None:
        if configs['runtime']['log_level_general'] == "DEBUG":
            verbose = 1
        else:
            verbose = 0

    # loading saved model
    model_arch_file = filename_no_ext + ".json"
    model_weights_file = filename_no_ext + ".h5"

    with open(model_arch_file, 'r') as arch_file:
        arch_json = arch_file.read()

    model = model_from_json(arch_json)
    logger.debug('Loading model weights.')
    model.load_weights(model_weights_file)
    logger.info('Model loaded correctly.')

    adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # load the data
    x_test = data_set.test.images
    y_test = data_set.test.labels

    # convert class vectors to binary class matrices
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # data_format = keras.backend.image_data_format()
    # reshaping it according to Theano rule
    # Theano backend uses (nb_sample, channels, height, width)

    x_test = reshape_images(x_test)

    logger.info('Loading test dataset for prediction.')

    logger.debug('x_test shape: {0}'.format(x_test.shape))
    logger.debug('Test samples: {0}'.format(x_test.shape[0]))

    x_test = x_test.astype('float32')

    # normalize each image separately
    if normalize:
        x_test = normalize_images(x_test)

    # convert class vectors to binary class matrices
    # y_test = np_utils.to_categorical(y_test, nb_classes)
    logger.info('Loading and formatting of data completed.')

    if configs['runtime']['log_level_general'] == "DEBUG":
        model.summary()

    logger.info('Predicting...')

    # compiling and calculating the score of the model again
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)

    if show_model_acc:
        logger.info('Model score: {0} {1}%'.format(model.metrics_names[1], score[1] * 100))

    if predict_probabilities:
        prob_predictions, uncertainty = predict_with_uncertainty(x_test, model, n_iter=mc_samples)

    # predicting the labels of the test set
    y_pred = model.predict_classes(x_test, batch_size=batch_size, verbose=verbose)

    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    np.set_printoptions(precision=2)
    logger.info('Confusion matrix, without normalization: ')
    logger.info(conf_matrix)

    target_pred_class = np.argmax(prob_predictions, axis=1).tolist()
    # predictions are an (n, m) array where
    # n: # of samples, m: # of classes
    # create dataframe with results
    df_cols = ['target_pred_class']
    for idx in range(prob_predictions.shape[1]):
        df_cols.append('prob_predictions_' + str(idx))
    df_cols.append('num_labels')
    df_cols.append('class_labels')

    # make a dataframe with the results and write it to file
    df_results = pd.DataFrame(np.column_stack((target_pred_class, prob_predictions, numerical_labels, text_labels)),
                              columns=df_cols)

    df_results.to_csv(results_file, index=False)

    # transform it in a list of n strings to be used by the viewer
    target_pred_probs = [str(['p' + str(i) + ':{0:.4f} '.format(item[i]) for i in range(nb_classes)]) for item in
                         prob_predictions]

    # insert new line if string too long
    # for item in target_pred_probs:
    #    item = insert_newlines(item, every=10)

    text_labels = text_labels.tolist()

    unique_class_labels = sorted(list(set(text_labels)))

    if plot_conf_matrix:
        logger.info("Calculating confusion matrix.")
        plot_confusion_matrix(conf_matrix, conf_matrix_file=conf_matrix_file, classes=unique_class_labels,
                              normalize=False, title='Confusion matrix')

    return target_pred_class, target_pred_probs, prob_predictions, conf_matrix, uncertainty


def predict_with_uncertainty(data, model, model_type='classification', n_iter=1000):
    """This function allows to calculate the uncertainty of a neural network model using dropout.

    This follows Chap. 3 in Yarin Gal's PhD thesis:
    http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf

    We calculate the uncertainty of the neural network predictions in the three ways proposed in Gal's PhD thesis,
     as presented at pag. 51-54:
    - variation_ratio: defined in Eq. 3.19
    - predictive_entropy: defined in Eq. 3.20
    - mutual_information: defined at pag. 53 (no Eq. number)

    Note: the current implementation works only for neural networks with layers that have the same behaviour
    at training and test time. For example, fully connected layers and convolution layers are allowed,
    while batch normalization layers are not.

    This is because the whole behaviour of the network is changed to the training learning phase, so that
    the dropout is used by Keras.

    Keras will soon implement a Dropout which can be kept also at test time; when this will be release, the function
    ``f`` below will no longer be necessary.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    # for some model with dropout ...
    f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    logger.info("Calculating classification uncertainty.")

    labels = []
    results = []
    for idx_iter in range(n_iter):
        if (idx_iter % (int(n_iter) / 10 + 1)) == 0:
            logger.info("Performing forward pass: {0}/{1}".format(idx_iter + 1, n_iter))

        result = f((data, 1))[0]
        label = np.argmax(result, axis=1)
        labels.append(label)
        results.append(result)

    results = np.asarray(results)
    prediction = results.mean(axis=0)

    if model_type == 'regression':
        predictive_variance = results.var(axis=0)
        uncertainty = dict(predictive_variance=predictive_variance)

    elif model_type == 'classification':
        # variation ratio
        mode, mode_count = stats.mode(np.asarray(labels))
        variation_ratio = np.transpose(1. - mode_count.mean(axis=0) / float(n_iter))

        # predictive entropy
        log_p_class = np.log2(prediction)
        entropy_all_iteration = - np.multiply(prediction, log_p_class)
        predictive_entropy = np.sum(entropy_all_iteration, axis=1)

        # mutual information
        p_log_p_all = np.multiply(np.log2(results), results)
        exp_p_omega = np.sum(np.sum(p_log_p_all, axis=0), axis=1)
        mutual_information = predictive_entropy + 1. / float(n_iter) * exp_p_omega

        uncertainty = dict(variation_ratio=variation_ratio, predictive_entropy=predictive_entropy,
                           mutual_information=mutual_information)
    else:
        raise ValueError("Supported model types are 'classification' or 'regression'."
                         "model_type={} is not accepted.".format(model_type))

    return prediction, uncertainty


def reshape_images(images):
    input_dims = (images.shape[1], images.shape[2])
    # works only for Keras 1.
    if len(input_dims) == 2:
        # add channels
        if keras.backend.image_dim_ordering() == 'th':
            images = np.reshape(images, (images.shape[0], -1, images.shape[1], images.shape[2]))
        elif keras.backend.image_dim_ordering() == 'tf':
            raise NotImplementedError('Tensorflow backend is not supported.')
        else:
            raise ValueError('Image ordering type not recognized. Possible values are th or tf.')
    else:
        raise Exception("Wrong number of dimensions.")

    return images


def normalize_images(images):
    input_dims = (images.shape[1], images.shape[2])

    if len(input_dims) == 2:
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = (images[idx, :, :, :] - np.amin(images[idx, :, :, :])) / (
                    np.amax(images[idx, :, :, :]) - np.amin(images[idx, :, :, :]))
    else:
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = (images[idx, :, :, :, :] - np.amin(images[idx, :, :, :, :])) / (
                    np.amax(images[idx, :, :, :, :]) - np.amin(images[idx, :, :, :, :]))

    return images
