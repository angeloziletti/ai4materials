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

from keras import backend as K
K.set_image_data_format("channels_last")

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from ai4materials.utils.utils_plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from scipy import stats
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import defaultdict

logger = logging.getLogger('ai4materials')


def train_neural_network(x_train, y_train, x_val, y_val, configs, partial_model_architecture, batch_size=32, nb_epoch=5,
                         normalize=True, checkpoint_dir=None, neural_network_name='my_neural_network',
                         training_log_file='training.log', early_stopping=False, data_augmentation=True):
    """Train a neural network to classify crystal structures represented as two-dimensional diffraction fingerprints.

    This model was introduced in [1]_.

    x_train: np.array, [batch, width, height, channels]


    .. [1] A. Ziletti, A. Leitherer, M. Scheffler, and L. M. Ghiringhelli,
        “Crystal-structure identification via Bayesian deep learning: towards superhuman performance”,
        in preparation (2018)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    if checkpoint_dir is None:
        checkpoint_dir = configs['io']['results_folder']

    filename_no_ext = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, neural_network_name)))

    training_log_file_path = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, training_log_file)))

    # reshape to follow the image conventions
    # - TensorFlow backend: [batch, width, height, channels]
    # - Theano backend: [batch, channels, width, height]
    # x_train = reshape_images_to_theano(x_train)
    # x_val = reshape_images_to_theano(x_val)

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

    if not data_augmentation:
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_val, y_val),
                  shuffle=True, verbose=0, callbacks=callbacks)
    else:
        logger.info('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                     samplewise_center=False,  # set each sample mean to 0
                                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                     samplewise_std_normalization=False,  # divide each input by its std
                                     zca_whitening=False,  # apply ZCA whitening
                                     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                     shear_range=0.0,  # value in radians, equivalent to 20 deg
                                     zoom_range=0.1,  # zoom_range = [1/1, 1],   #same as in NIPS 2015 paper.
                                     width_shift_range=4.0,  # randomly shift images horizontally
                                     height_shift_range=4.0,  # randomly shift images vertically
                                     horizontal_flip=False,  # randomly flip images
                                     vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        # Not required as it is Only required if featurewise_center or featurewise_std_normalization or zca_whitening.
        #        datagen.fit(x_train)

        # fit the model on the batches generated by datagen.flow() and save the loss and acc data history in the hist variable
        # filepath = "/home/310251680/work/scripts/imageCLEF_reprod/saved_models/model_imgCLEF_shallow_ep_10_weights.hdf5"
        # save_model_per_epoch = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto')

        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      samples_per_epoch=x_train.shape[0], nb_epoch=nb_epoch,
                                      validation_data=(x_val, y_val), callbacks=callbacks, verbose=0)

    # serialize weights to HDF5
    # model.save(filename_no_ext + ".h5")
    # logger.info("Model saved to disk.")
    # logger.info("Filename: {0}".format(filename_no_ext))
    del model


def predict(x, y, configs, numerical_labels, text_labels, nb_classes=3, results_file=None, model=None, batch_size=32,
            conf_matrix_file=None, verbose=1, with_uncertainty=True, mc_samples=50, 
            consider_memory=True, max_length=1e6):
    uncertainty = None

    if results_file is None:
        results_file = configs['io']['results_file']

    if conf_matrix_file is None:
        conf_matrix_file = configs['io']['conf_matrix_file']

    # convert class vectors to binary class matrices - one-hot encoding
    y = np_utils.to_categorical(y, nb_classes)

    # get the input shape from the Keras model
    # the first element in the list is the batch size (not defined yet, it is a placeholder)
    # we keep all the indices after the first one to correctly reshape our array
    input_shape_from_model = model.layers[0].get_input_at(0).get_shape().as_list()[1:]

    # add -1 so that we can reshape properly for any number of images (or sequences) in the dataset
    target_shape = tuple([-1] + input_shape_from_model)

    x = reshape_images(x, target_shape=target_shape)
    x = x.astype('float32')

    logger.info('Loading test dataset for prediction.')
    logger.debug('x_test shape: {0}'.format(x.shape))
    logger.debug('Test samples: {0}'.format(x.shape[0]))
    logger.info('Loading and formatting of data completed.')

    if configs['runtime']['log_level_general'] == "DEBUG":
        model.summary()

    logger.info('Predicting...')

    if with_uncertainty:
        if consider_memory:
            logger.info("While considering memory limits: Using multiple passes to have principles probability and uncertainty estimates")
            uncertainty = defaultdict(list)
            prob_predictions = np.array([])
            current_size = mc_samples*x.shape[0] # x.shape[0]=batch_size
            if current_size>=max_length:
                split_size = 1
                # increase split size until  we are safe to not run into memory problems
                while (current_size/split_size)>=max_length:
                    split_size+=1
                x_splitted = np.array_split(x, split_size)
                logger.info("Had to split input of shape {} into {} subarrays".format(x.shape, split_size))
                for data, split in zip(x_splitted, np.arange(len(x_splitted))+1):
                    logger.info("Split # {} / {}".format(split,split_size))
                    prediction_, uncertainty_ = predict_with_uncertainty(data, model=model, n_iter=mc_samples)
                    if split==1:
                        prob_predictions = prediction_
                    else:
                        prob_predictions = np.concatenate((prob_predictions, prediction_), axis=0)
                    for key_ in uncertainty_:
                        uncertainty[key_].extend(uncertainty_[key_])
            else:
                logger.info("Using multiple passes to have principles probability and uncertainty estimates")
                prob_predictions, uncertainty = predict_with_uncertainty(x, model, n_iter=mc_samples)                
        else:
            logger.info("Using multiple passes to have principles probability and uncertainty estimates")
            prob_predictions, uncertainty = predict_with_uncertainty(x, model, n_iter=mc_samples)
    else:
        logger.info("Using only a single pass. No uncertainty estimation.")
        prob_predictions = model.predict(x, batch_size=batch_size, verbose=verbose)

    # get the argmax to have the class label from the probabilities
    y_pred = prob_predictions.argmax(axis=-1)
    logger.info("Accuracy: {}%".format(accuracy_score(np.argmax(y, axis=1), y_pred) * 100.))

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
    # df_cols.append('predictive')

    data = np.column_stack((target_pred_class, prob_predictions, numerical_labels, text_labels))

    # make a dataframe with the results and write it to file
    if uncertainty is not None:
        for key, value in uncertainty.items():
            data = np.column_stack((data, value))
            df_cols.append('uncertainty_' + str(key))

    df_results = pd.DataFrame(data=data, columns=df_cols)
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
                   string_probs=string_probs, uncertainty=uncertainty)

    return results


def predict_with_uncertainty(data, model, model_type='classification', n_iter=1000):
    """This function allows to calculate the uncertainty of a neural network model using dropout.

    This follows Chap. 3 in Yarin Gal's PhD thesis:
    http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf

    We calculate the uncertainty of the neural network predictions in the three ways proposed in Gal's PhD thesis,
     as presented at pag. 51-54:
    - variation_ratio: defined in Eq. 3.19
    - predictive_entropy: defined in Eq. 3.20
    - mutual_information: defined at pag. 53 (no Eq. number)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    logger.info("Calculating classification uncertainty.")

    labels = []
    results = []
    for idx_iter in range(n_iter):
        if (idx_iter % (int(n_iter) / 10 + 1)) == 0:
            logger.info("Performing forward pass: {0}/{1}".format(idx_iter + 1, n_iter))

        result = model.predict(data)
        label = result.argmax(axis=-1)

        labels.append(label)
        results.append(result)
        
    results = np.asarray(results)
    # less memory intensive:
    #if len(results)>25000*1000:
    #    dtype_ = np.float16
    #else:
    #    dtype_ = None
    #results = np.asarray(results, dtype=dtype_)
    prediction = results.mean(axis=0)

    if model_type == 'regression':
        predictive_variance = results.var(axis=0)
        uncertainty = dict(predictive_variance=predictive_variance)

    elif model_type == 'classification':
        # variation ratio
        #print(labels)
        mode, mode_count = stats.mode(np.asarray(labels))
        #print(mode)
        #print(mode_count)
        variation_ratio = np.transpose(1. - mode_count.mean(axis=0) / float(n_iter))
        #print(variation_ratio)
        # predictive entropy
        # clip values to 1e-12 to avoid divergency in the log
        prediction = np.clip(prediction, a_min=1e-12, a_max=None, out=prediction)
        log_p_class = np.log2(prediction)
        entropy_all_iteration = - np.multiply(prediction, log_p_class)
        predictive_entropy = np.sum(entropy_all_iteration, axis=1)

        # mutual information
        # clip values to 1e-12 to avoid divergency in the log
        results = np.clip(results, a_min=1e-12, a_max=None, out=results)
        p_log_p_all = np.multiply(np.log2(results), results)
        exp_p_omega = np.sum(np.sum(p_log_p_all, axis=0), axis=1)
        mutual_information = predictive_entropy + 1. / float(n_iter) * exp_p_omega

        uncertainty = dict(variation_ratio=variation_ratio, predictive_entropy=predictive_entropy,
                           mutual_information=mutual_information)
    else:
        raise ValueError("Supported model types are 'classification' or 'regression'."
                         "model_type={} is not accepted.".format(model_type))

    return prediction, uncertainty


def reshape_images(images, target_shape):
    """Reshape images according to the target shape"""

    images = np.reshape(images, target_shape)
    logger.debug("Reshaping to {}".format(target_shape))

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
