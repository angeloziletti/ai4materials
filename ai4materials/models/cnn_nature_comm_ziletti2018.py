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
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ai4materials.utils.utils_plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from ai4materials.utils.utils_config import get_data_filename
from ai4materials.utils.utils_neural_networks import load_model
import os
import logging
import pandas as pd
import numpy as np

K.set_image_dim_ordering('th')

logger = logging.getLogger('ai4materials')


def train_neural_network(x_train, y_train, x_val, y_val, configs, partial_model_architecture, batch_size=32, nb_epoch=5,
                         normalize=True, checkpoint_dir=None, neural_network_name='my_neural_network',
                         training_log_file='training.log', early_stopping=False):
    """Train a neural network to classify crystal structures represented as two-dimensional diffraction fingerprints.

    This model was introduced in [1]_.

    x_train: np.array, [batch, width, height, channels]


    .. [1] A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli,
        “Insightful classification of crystal structures using deep learning”,
        Nature Communications, vol. 9, pp. 2775 (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    if checkpoint_dir is None:
        checkpoint_dir = configs['io']['results_folder']

    filename_no_ext = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, neural_network_name)))

    training_log_file_path = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, training_log_file)))

    # reshape to follow the image conventions
    # - TensorFlow backend: [batch, width, height, channels]
    # - Theano backend: [batch, channels, width, height]
    x_train = reshape_images_to_theano(x_train)
    x_val = reshape_images_to_theano(x_val)

    assert x_train.shape[1] == x_val.shape[1]
    assert x_train.shape[2] == x_val.shape[2]
    assert x_train.shape[3] == x_val.shape[3]

    img_channels = x_train.shape[1]
    img_width = x_train.shape[2]
    img_height = x_train.shape[3]

    logger.info('Loading datasets.')
    logger.debug('x_train shape: {0}'.format(x_train.shape))
    logger.debug('y_train shape: {0}'.format(y_train.shape))
    logger.debug('x_val shape: {0}'.format(x_val.shape))
    logger.debug('y_val shape: {0}'.format(y_val.shape))
    logger.debug('Training samples: {0}'.format(x_train.shape[0]))
    logger.debug('Validation samples: {0}'.format(x_val.shape[0]))
    logger.debug("Img channels: {}".format(x_train.shape[1]))
    logger.debug("Img width: {}".format(x_train.shape[2]))
    logger.debug("Img height: {}".format(x_train.shape[3]))

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    # normalize each image separately
    if normalize:
        for idx in range(x_train.shape[0]):
            x_train[idx, :, :, :] = (x_train[idx, :, :, :] - np.amin(x_train[idx, :, :, :])) / (
                    np.amax(x_train[idx, :, :, :]) - np.amin(x_train[idx, :, :, :]))
        for idx in range(x_val.shape[0]):
            x_val[idx, :, :, :] = (x_val[idx, :, :, :] - np.amin(x_val[idx, :, :, :])) / (
                    np.amax(x_val[idx, :, :, :]) - np.amin(x_val[idx, :, :, :]))

    # check if the image is already normalized
    logger.info(
        'Maximum value in x_train for the 1st image (to check normalization): {0}'.format(np.amax(x_train[0, :, :, :])))
    logger.info(
        'Maximum value in x_val for the 1st image (to check normalization): {0}'.format(np.amax(x_val[0, :, :, :])))

    # convert class vectors to binary class matrices
    nb_classes = len(set(y_train))
    nb_classes_val = len(set(y_val))

    if nb_classes_val != nb_classes:
        raise ValueError("Different number of unique classes in training and validation set: {} vs {}."
                         "Training set unique classes: {}"
                         "Validation set unique classes: {}".format(nb_classes, nb_classes_val, set(y_train),
                                                                    set(y_val)))

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_val = np_utils.to_categorical(y_val, nb_classes)

    logger.info('Loading and formatting of data completed.')

    # return the Keras model
    model = partial_model_architecture(n_rows=img_width, n_columns=img_height, img_channels=img_channels,
                                       nb_classes=nb_classes)

    model.summary()
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

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_val, y_val), shuffle=True,
              verbose=0, callbacks=callbacks)

    # serialize weights to HDF5
    model.save(filename_no_ext + ".h5")
    logger.info("Model saved to disk.")
    logger.info("Filename: {0}".format(filename_no_ext))
    del model


def predict(x, y, configs, numerical_labels, text_labels, nb_classes=7, results_file=None, model=None, batch_size=32,
            show_model_acc=True, conf_matrix_file=None, verbose=1):
    """Predict the class of crystal structures represented with the two-dimensional diffraction fingerprints.


    This model was introduced in Ref. [1]_:


    Returns:

    ``keras.models.Model`` object
        Return the Keras model from the reference mentioned above.

    .. [1] A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli,
        “Insightful classification of crystal structures using deep learning”,
        Nature Communications, vol. 9, pp. 2775 (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    if model is None:
        model = load_nature_comm_ziletti2018_network()

    if results_file is None:
        results_file = configs['io']['results_file']

    if conf_matrix_file is None:
        conf_matrix_file = configs['io']['conf_matrix_file']

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # convert class vectors to binary class matrices - one-hot encoding
    y = np_utils.to_categorical(y, nb_classes)

    # reshaping it according to Theano rule
    # Theano backend uses (nb_sample, channels, height, width)
    x = reshape_images_to_theano(x)
    x = x.astype('float32')

    logger.info('Loading test dataset for prediction.')
    logger.debug('x_test shape: {0}'.format(x.shape))
    logger.debug('Test samples: {0}'.format(x.shape[0]))
    logger.info('Loading and formatting of data completed.')

    if configs['runtime']['log_level_general'] == "DEBUG":
        model.summary()

    logger.info('Predicting...')

    # compiling and calculating the score of the model again
    score = model.evaluate(x, y, batch_size=batch_size, verbose=verbose)

    if show_model_acc:
        logger.info('Model score: {0} {1}%'.format(model.metrics_names[1], score[1] * 100))

    prob_predictions = model.predict(x, batch_size=batch_size, verbose=verbose)

    # predicting the labels of the test set
    y_pred = model.predict_classes(x, batch_size=batch_size, verbose=verbose)

    conf_matrix = confusion_matrix(np.argmax(y, axis=1), y_pred)
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
    logger.info("Predictions written to: {}".format(results_file))

    text_labels = text_labels.tolist()
    unique_class_labels = sorted(list(set(text_labels)))

    plot_confusion_matrix(conf_matrix, conf_matrix_file=conf_matrix_file, classes=unique_class_labels, normalize=False,
                          title='Confusion matrix')
    logger.info("Confusion matrix written to {}.".format(conf_matrix_file))

    # transform it in a list of n strings to be used by the viewer
    string_probs = [str(['p' + str(i) + ':{0:.4f} '.format(item[i]) for i in range(nb_classes)]) for item in
                    prob_predictions]

    # insert new line if string too long
    # for item in target_pred_probs:
    #    item = insert_newlines(item, every=10)

    results = dict(target_pred_class=target_pred_class, prob_predictions=prob_predictions, confusion_matrix=conf_matrix,
                   string_probs=string_probs)

    return results


def load_nature_comm_ziletti2018_network():
    """Load the Keras neural network model to classify two-dimensional diffraction fingerprints introduced in Ref. [1]_


    Returns:

    ``keras.models.Model`` object
        Return the Keras model from the reference mentioned above.

    .. [1] A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli,
        “Insightful classification of crystal structures using deep learning”,
        Nature Communications, vol. 9, pp. 2775 (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    # rgb convolutional neural network
    model_arch_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.json')
    model_weights_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.h5')

    model = load_model(model_arch_file, model_weights_file)

    return model


def reshape_images_to_theano(images):
    # works only for Keras 1.

    if keras.backend.image_dim_ordering() == 'th':
        if len(images.shape) == 4:
            # add channels
            images = np.reshape(images, (images.shape[0], -1, images.shape[1], images.shape[2]))
        elif len(images.shape) == 3:
            images = np.reshape(images, (images.shape[0], 1, images.shape[1], images.shape[2]))
        else:
            raise Exception("Wrong number of dimensions."
                            "Images' dimensions: {}".format(images.shape))
    elif keras.backend.image_dim_ordering() == 'tf':
        raise NotImplementedError('Tensorflow backend is not supported.')
    else:
        raise ValueError('Image ordering type not recognized. Possible values are th or tf.')

    return images


def normalize_images(images):
    if len(images.shape) == 4:
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = (images[idx, :, :, :] - np.amin(images[idx, :, :, :])) / (
                    np.amax(images[idx, :, :, :]) - np.amin(images[idx, :, :, :]))
    else:
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = (images[idx, :, :, :, :] - np.amin(images[idx, :, :, :, :])) / (
                    np.amax(images[idx, :, :, :, :]) - np.amin(images[idx, :, :, :, :]))

    return images
