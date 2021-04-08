# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:02:51 2018

@author: leitherer
"""

import matplotlib

matplotlib.use('Agg')  # This way do not show plot windows when compute SOAP and FT-SOAP
import matplotlib.pyplot as plt

import os.path

import numpy as np
from sklearn.metrics import roc_auc_score

from argparse import ArgumentParser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform, choice
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
import pickle

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import sys


def data():
    num_classes = 5
    dataset_folder = "/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets"
    test_split = 0.1

    def load_obj(dataset_folder, name):
        with open(os.path.join(dataset_folder, name) + '.pkl', 'rb') as f:
            return pickle.load(f, encoding='bytes')

    pristine_x = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_pristine_x')
    pristine_y = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_pristine_y')

    displacement_2_x = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_displacement-2%_x')
    displacement_2_y = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_displacement-2%_y')

    displacement_4_x = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_displacement-4%_x')
    displacement_4_y = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_displacement-4%_y')

    vacancies_25_x = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_vacancies-25%_x')
    vacancies_25_y = load_obj(dataset_folder, 'hcp-sc-fcc-diam-bcc_vacancies-25%_y')

    numerical_to_text_label = {0: 'hcp', 1: 'sc', 2: 'fcc', 3: 'diam', 4: 'bcc'}

    # Get class labels
    classes = []
    for i in numerical_to_text_label.keys():
        classes.append(numerical_to_text_label[i])

    pristine_y_ohe = keras.utils.to_categorical(pristine_y, num_classes=num_classes)

    # Split into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(pristine_x, pristine_y_ohe, test_size=test_split,
                                                        random_state=4, stratify=pristine_y_ohe)

    # print("NB: using displacements as validation set.")
    x_test = displacement_4_x
    y_test = displacement_4_y
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

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

    datagen.fit(x_test)

    return datagen, x_train, y_train, x_test, y_test


def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    nb_classes = 5
    datagen, x_train, y_train, x_val, y_val = data()

    global ITERATION
    ITERATION += 1

    print ('Params testing: ', params)

    model = Sequential()
    model.add(Conv2D(params['nb_filters_conv1'], (params['k_size_conv1'], params['k_size_conv1']), name="convolution2d_1", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True, input_shape=(52, 32, 1)))

    model.add(Conv2D(params['nb_filters_conv2'], (params['k_size_conv2'], params['k_size_conv2']), name="convolution2d_2", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_1'))

    model.add(Conv2D(params['nb_filters_conv3'], (params['k_size_conv3'], params['k_size_conv3']), name="convolution2d_3", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True))
    model.add(Conv2D(params['nb_filters_conv4'], (params['k_size_conv4'], params['k_size_conv4']), name="convolution2d_4", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpooling2d_2'))

    model.add(Conv2D(params['nb_filters_conv5'], (params['k_size_conv5'], params['k_size_conv5']), name="convolution2d_5", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True))
    model.add(Conv2D(params['nb_filters_conv6'], (params['k_size_conv6'], params['k_size_conv6']), name="convolution2d_6", activation="relu", padding="same",
                     kernel_initializer="orthogonal", use_bias=True))

    model.add(Flatten(name='flatten_1'))
    model.add(BatchNormalization())
    model.add(Dense(units=params['hidden_units'], name="dense_1", activation="relu", use_bias=True))
    model.add(Dropout(params['dropout'], name='dropout_1'))
    model.add(Dense(nb_classes, name='dense_2'))
    model.add(Activation('softmax', name='activation_1'))

    # model.summary()
    # compile model
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=params['batch_size']),
                        steps_per_epoch=int(x_train.shape[0] / params['batch_size']),
                        # nb_epoch=params['nb_epoch'], validation_data=(x_val, y_val))
    epochs=1, validation_data = (x_val, y_val))

    score, acc = model.evaluate(x_val, y_val, verbose=0)

    # Write to the csv file ('a' means append)
    of_connection = open(outfile, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([ITERATION, acc,
                     params['nb_filters_conv1'], params['nb_filters_conv2'], params['nb_filters_conv3'],
                     params['nb_filters_conv3'], params['nb_filters_conv5'], params['nb_filters_conv6'],
                     params['k_size_conv1'], params['k_size_conv2'], params['k_size_conv3'],
                     params['k_size_conv3'], params['k_size_conv5'], params['k_size_conv6'],
                     params['dropout'], params['hidden_units'], params['batch_size']])


    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":

    #  Set up folders
    ########################################

    parser = ArgumentParser()
    parser.add_argument("-m", "--machine", dest="machine", help="on which machine the script is run", metavar="MACHINE")
    args = parser.parse_args()

    machine = vars(args)['machine']

    # machine = 'draco'
    machine = 'local'

    if machine == 'draco':
        main_folder = "/ptmp/ziang/rot_inv_3d/"
        savepath_model = '/ptmp/ziang/rot_inv_3d/saved_models'
        dataset_folder = '/ptmp/ziang/rot_inv_3d/datasets'
    elif machine == 'local':
        main_folder = "/home/ziletti/Documents/calc_nomadml/rot_inv_3d/"
        savepath_model = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models'
        dataset_folder = "/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets"

    ################################
    # Load datasets
    ################################

    import pickle

    import keras
    from keras.layers import concatenate, Dense, Dropout
    from keras.models import Model

    #
    space = {'nb_filters_conv1': hp.choice('nb_filters_conv1', [8, 16, 32, 64, 128]),
             'nb_filters_conv2': hp.choice('nb_filters_conv2', [8, 16, 32, 64, 128]),
             'nb_filters_conv3': hp.choice('nb_filters_conv3', [8, 16, 32, 64, 128]),
             'nb_filters_conv4': hp.choice('nb_filters_conv4', [8, 16, 32, 64, 128]),
             'nb_filters_conv5': hp.choice('nb_filters_conv5', [8, 16, 32, 64, 128]),
             'nb_filters_conv6': hp.choice('nb_filters_conv6', [8, 16, 32, 64, 128]),
             'k_size_conv1': hp.choice('k_size_conv1', [3, 5]),
             'k_size_conv2': hp.choice('k_size_conv2', [3, 5]),
             'k_size_conv3': hp.choice('k_size_conv3', [3, 5]),
             'k_size_conv4': hp.choice('k_size_conv4', [3, 5]),
             'k_size_conv5': hp.choice('k_size_conv5', [3, 5]),
             'k_size_conv6': hp.choice('k_size_conv6', [3, 5]),
             'dropout': hp.uniform('dropout', .05, .25),
             'hidden_units': hp.choice('hidden_units', [32, 64, 128, 256, 512]),
             'batch_size': hp.choice('batch_size', [32, 64, 128])#,
             # 'nb_epoch': hp.choice('nb_epoch', [100, 200, 500]),
             # 'lr': hp.uniform('lr', .1, 0.0001)
             }

    import csv

    # Global variable
    global ITERATION

    ITERATION = 0

    outfile = os.path.join(main_folder, 'optimization.csv')

    # File to save first results
    of_connection = open(outfile, 'w')
    writer = csv.writer(of_connection)

    # Write the headers to the file
    writer.writerow(['iteration', 'accuracy',
                     'nb_filters_conv1', 'nb_filters_conv2', 'nb_filters_conv3', 'nb_filters_conv4',
                     'nb_filters_conv5', 'nb_filters_conv6',
                     'k_size_conv1', 'k_size_conv2', 'k_size_conv3', 'k_size_conv4', 'k_size_conv5', 'k_size_conv6',
                     'dropout', 'hidden_units', 'batch_size'])

    of_connection.close()

    trials = Trials()
    best = fmin(f_nn, space, algo=tpe.suggest, max_evals=2, trials=trials)
    print('best: ')
    print(space_eval(space, best))
