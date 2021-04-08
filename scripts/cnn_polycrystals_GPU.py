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

from argparse import ArgumentParser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


def cnn_architecture_polycrystals(learning_rate, conv2d_filters, kernel_sizes, hidden_layer_size, n_rows, n_columns,
                      img_channels, nb_classes, dropout, plot_the_model=False, model_name='cnn_polycrystals'):
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
        Convolution2D(conv2d_filters[0], kernel_sizes[0], kernel_sizes[0], name='convolution2d_1', activation='relu',
               border_mode='same', init='orthogonal', bias=True, input_shape=(n_rows, n_columns, img_channels)))
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
    # model.add(BatchNormalization())
    # model.add(Dense(hidden_layer_size, name='dense_1', activation='relu', bias=True))
    # model.add(Dropout(dropout, name='dropout_1'))
    model.add(Dense(nb_classes, name='dense_2'))
    model.add(Activation('softmax', name='activation_1'))

    # plot model - may crash on draco
    if plot_the_model:
        plot_model(model, to_file=savepath_model + '/' + model_name + '.png', show_shapes=True, show_layer_names=True)

    model.summary()
    # compile model
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    return model


def train_and_test_model(descriptor_name, model, savepath_model, batch_size, epochs, test_split, pristine_x, pristine_y,
                         vacancies_x, vacancies_y, displacements_x, displacements_y, num_classes=5,
                         numerical_to_text_label=None, verbose=2, data_augmentation=False):

    """
    Function for training a given model and testing it on defective structures (vacancies and displacements).

    descriptor_name: string
        Name of the descriptor (soap, ft-soap), that is used.

    model: Keras model object
        Model to be trained and tested.

    savepath_model:
        Path to which the model is saved.

    batch_size: int
        Batch size used for training and testing the model.
        If batch_size='max', only one batch containing all of the training/test data is used.

    epochs: int
        Number of epochs used for training the model.

    test_split: float
        Split percentage for train/validation split.

    pristine_x, pristine_y: each 1D lists
        Descriptors and labels for pristine structures.
        Pristine_y should contain only numerical labels, which is checked
        by assert statements (one hot encoding is done by default)

    vacancies_x, vacancies_y, displacements_x, displacements_y: each 1D lists
        Descriptors and labels for defective structures.

    num_classes: int
        Number of classes.

    numerical_to_text_label: Dictionary
        Dictionary for conversion of numerical to text labels.

    verbose: int (0-2)
        Sets the verbosity mode (verbose=2 prints maximum info to terminal).
        See https://stackoverflow.com/questions/46218407/how-to-interpret-keras-model-fit-output.

    only_one_GRU_cell: bool
        If True, then validation data will be reshaped according to this
        specific model architecture of one GRU cell.

    Returns:

        the arrays y_pred_vac, y_true_vac, y_pred_displ, y_true_displ
        that contain, in one hot encoded format:
            - the predictions (vacancies: y_pred_vac, displacements: y_pred_displ)
            - the true labels (vacancies: y_true_vac, displacements: y_true_displ)

    """

    # Get class labels
    classes = []
    for i in numerical_to_text_label.keys():
        classes.append(numerical_to_text_label[i])

    pristine_y_ohe = keras.utils.to_categorical(pristine_y, num_classes=num_classes)

    # Split into random train and test subsets
    x_train, x_test, y_train, y_test = train_test_split(pristine_x, pristine_y_ohe, test_size=test_split,
                                                        random_state=4, stratify=pristine_y_ohe)

    print("NB: using displacements as validation set.")
    x_test = displacements_x
    y_test = displacements_y
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model_name = 'model'
    callbacks = []
    save_model_per_epoch = ModelCheckpoint(savepath_model + '/' + model_name + ".h5",
                                           monitor='val_categorical_accuracy', verbose=1,
                                           save_best_only=True, mode='max', period=1)
    callbacks.append(save_model_per_epoch)

    model.summary()

    if not data_augmentation:
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(x_test, y_test), callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')

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


        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      samples_per_epoch=x_train.shape[0], epochs=epochs,
                                      validation_data=(x_test, y_test), callbacks=callbacks, verbose=verbose)



    # summarize history for accuracy: A plot of accuracy on the training and validation datasets over training epochs.
    # From https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(savepath_model + '/' + descriptor_name + '_acc_and_val_acc_over_epochs.png')
    plt.close()
    # summarize history for loss: A plot of loss on the training and validation datasets over training epochs.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(savepath_model + '/' + descriptor_name + '_loss_on_training_and_validation_data_over_epochs.png')
    plt.close()

    # Test model
    if batch_size == 'max':
        batch_size = x_test.shape[0]
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # Validate model
    vac_val_data_x = vacancies_x.reshape(vacancies_x.shape[0], vacancies_x.shape[1], 1)
    vac_val_data_y = vacancies_y  # true labels - not o.h.e!

    if batch_size == 'max':
        batch_size = vacancies_x.shape[0]
    vac_class_predictions_prob = model.predict(vac_val_data_x, batch_size=batch_size, verbose=0)
    vac_class_predictions = vac_class_predictions_prob.argmax(axis=-1)
    # Explanation of argmax(axis=-1): https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
    # " means that the index that will be returned by argmax will be taken from the last axis. "

    conf_mat = confusion_matrix(y_true=vac_val_data_y, y_pred=vac_class_predictions)
    plot_confusion_matrix(conf_mat, classes, True, 'Confusion matrix for vacancies ' + str(numerical_to_text_label),
                          savepath_model + '/' + descriptor_name + '_vacancies_conf_mat.png', plt.cm.Blues)

    displ_val_data_x = displacements_x.reshape(displacements_x.shape[0], displacements_x.shape[1], 1)
    displ_val_data_y = displacements_y  # true labels - not ohe!

    displ_class_predictions_prob = model.predict(displ_val_data_x, batch_size=batch_size, verbose=0)
    displ_class_predictions = displ_class_predictions_prob.argmax(axis=-1)

    conf_mat = confusion_matrix(y_true=displ_val_data_y, y_pred=displ_class_predictions)
    plot_confusion_matrix(conf_mat, classes, True, 'Confusion matrix for displacements ' + str(numerical_to_text_label),
                          savepath_model + '/' + descriptor_name + '_displacements_conf_mat.png', plt.cm.Blues)

    model.save(savepath_model + '/' + descriptor_name + '_one_lstm_model.h5')

    # Return predictions
    y_pred_vac = keras.utils.to_categorical(vac_class_predictions, num_classes=num_classes)
    y_true_vac = keras.utils.to_categorical(vac_val_data_y, num_classes=num_classes)

    y_pred_displ = keras.utils.to_categorical(displ_class_predictions, num_classes=num_classes)
    y_true_displ = keras.utils.to_categorical(displ_val_data_y, num_classes=num_classes)

    return y_pred_vac, y_true_vac, y_pred_displ, y_true_displ


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', savefig_name='conf_mat.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
    plt.savefig(savefig_name)
    plt.close()

    ########################################

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        if machine == 'draco':
            return pickle.load(f, encoding='bytes')
        elif machine == 'local':
            # return pickle.load(f)
            return pickle.load(f, encoding='bytes')


if __name__ == "__main__":

    #  Set up folders
    ########################################

    parser = ArgumentParser()
    parser.add_argument("-m", "--machine", dest="machine", help="on which machine the script is run", metavar="MACHINE")
    args = parser.parse_args()

    machine = vars(args)['machine']

    #machine = 'draco'
    machine = 'local'

    if machine == 'draco':
        main_folder = "/ptmp/ziang/rot_inv_3d/"
        savepath_model = '/ptmp/ziang/rot_inv_3d/saved_models'
    elif machine == 'local':
        main_folder = "/home/ziletti/Documents/calc_nomadml/rot_inv_3d/"
        savepath_model = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models'

    ################################
    # Load datasets
    ################################

    import pickle

    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
    os.chdir(dataset_folder)

    pristine_x = load_obj('hcp-sc-fcc-diam-bcc_pristine_x')
    pristine_y = load_obj('hcp-sc-fcc-diam-bcc_pristine_y')

    displacement_2_x = load_obj('hcp-sc-fcc-diam-bcc_displacement-2%_x')
    displacement_2_y = load_obj('hcp-sc-fcc-diam-bcc_displacement-2%_y')

    displacement_4_x = load_obj('hcp-sc-fcc-diam-bcc_displacement-4%_x')
    displacement_4_y = load_obj('hcp-sc-fcc-diam-bcc_displacement-4%_y')

    vacancies_25_x = load_obj('hcp-sc-fcc-diam-bcc_vacancies-25%_x')
    vacancies_25_y = load_obj('hcp-sc-fcc-diam-bcc_vacancies-25%_y')

    numerical_to_text_label = {0: 'hcp', 1: 'sc', 2: 'fcc', 3: 'diam', 4: 'bcc'}

    import keras
    from keras.layers import concatenate, Dense, Dropout
    from keras.models import Model

    learning_rate = 0.001

    cnn_polycrystals = cnn_architecture_polycrystals(learning_rate=learning_rate,
                                                     conv2d_filters=[32, 32, 16, 16, 16, 16],
                                                     kernel_sizes=[3, 3, 3, 3, 3, 3],
                                                     hidden_layer_size=64, img_channels=1, nb_classes=5,
                                                     dropout=0.1, n_rows=pristine_x.shape[1],
                                                     n_columns=pristine_x.shape[2])

    # Train and test models
    y_pred_vac25, y_true_vac25, y_pred_displ2, y_true_displ2 = train_and_test_model(
        descriptor_name='soap', model=cnn_polycrystals, savepath_model=savepath_model, batch_size=128,
        epochs=1000, test_split=0.2, pristine_x=pristine_x, pristine_y=pristine_y,
        vacancies_x=vacancies_25_x, vacancies_y=vacancies_25_y, displacements_x=displacement_2_x,
        displacements_y=displacement_2_y, numerical_to_text_label=numerical_to_text_label, verbose=2,
        data_augmentation=True)






