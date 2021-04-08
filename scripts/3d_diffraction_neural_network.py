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
__copyright__ = "Copyright 2016-2018, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/04/18"

if __name__ == "__main__":
    import sys
    import os.path

    atomic_data_dir = os.path.normpath('/home/ziletti/nomad/nomad-lab-base/analysis-tools/atomic-data')

    sys.path.insert(0, atomic_data_dir)

    from argparse import ArgumentParser
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.descriptors.diffraction3d import DISH
    from ai4materials.models.cnn_polycrystals import predict
    from ai4materials.models.cnn_architectures import cnn_architecture_polycrystals
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from datetime import datetime
    from keras.models import load_model
    import numpy as np

    startTime = datetime.now()
    now = datetime.now()

    parser = ArgumentParser()
    parser.add_argument("-m", "--machine", dest="machine", help="on which machine the script is run", metavar="MACHINE")
    args = parser.parse_args()

    machine = vars(args)['machine']
    # machine = 'eos'

    if machine == 'eos':
        config_file = '/scratch/ziang/diff_3d/config_prototypes.yml'
        main_folder = '/scratch/ziang/diff_3d/'
        prototypes_basedir = '/scratch/ziang/diff_3d/prototypes_aflow_new/'
        db_files_prototypes_basedir = '/scratch/ziang/diff_3d/db_ase_prototypes'

    else:
        config_file = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/config_diff3d.yml'
        main_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/'
        prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/prototypes_aflow_new'
        db_files_prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase_prototypes'

    # read config file
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))

    configs['io']['dataset_folder'] = dataset_folder

    descriptor = DISH(configs=configs)

    train_set_name = 'hcp-sc-fcc-diam-bcc_pristine'
    path_to_x_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')))

    test_set_name = 'hcp-sc-fcc-diam-bcc_vacancies-1%'
    # test_set_name = 'hcp-sc-fcc-diam-bcc_displacement-20%'
    # test_set_name = 'hcp-sc-fcc-diam-bcc_pristine'
    path_to_x_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_x.pkl')))
    path_to_y_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_y.pkl')))
    path_to_summary_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_summary.json')))

    # load the data
    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                               path_to_summary=path_to_summary_test)

    params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
                  "batch_size": 32}

    text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

    # partial_model_architecture = partial(cnn_architecture_polycrystals, conv2d_filters=[32, 16, 8, 8, 16, 32],
    #                                  kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64, dropout=0.1)

    # train_neural_network(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, configs=configs,
    #                      partial_model_architecture=partial_model_architecture,
    #                      batch_size=params_cnn["batch_size"], checkpoint_dir=checkpoint_dir,
    #                      neural_network_name=params_cnn["checkpoint_filename"],
    #                      nb_epoch=20, training_log_file=training_log_file, early_stopping=False, normalize=True)

    # load trained neural network from hdf5 file
    path_to_saved_model = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models/enc_dec_drop12.5/model.h5'
    model = load_model(path_to_saved_model)

    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix_' + test_set_name + '.png')))

    results = predict(x=x_test, y=y_test, configs=configs, numerical_labels=numerical_labels, text_labels=text_labels,
                      nb_classes=params_cnn["nb_classes"], model=model, batch_size=params_cnn["batch_size"],
                      conf_matrix_file=conf_matrix_file, results_file=results_file)

    logger.info("Average predictive entropy: {}".format(np.mean(results['uncertainty']['predictive_entropy'])))
    logger.info("Average mutual information: {}".format(np.mean(results['uncertainty']['mutual_information'])))

    sys.exit()

    # sys.exit()

    model_weights_file = os.path.join(checkpoint_dir, '{0}.h5'.format(
        params_cnn['checkpoint_filename']))

    model_arch_file = os.path.join(checkpoint_dir, '{0}.json'.format(
        params_cnn['checkpoint_filename']))

    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    nb_imgs = 2
    images_classes = []
    images_class_0 = x_train[(y_train == 0)]
    images_class_1 = x_train[(y_train == 1)]
    images_class_2 = x_train[(y_train == 2)]
    images_class_3 = x_train[(y_train == 3)]
    images_class_4 = x_train[(y_train == 4)]

    images_classes.append(images_class_0)
    images_classes.append(images_class_1)
    images_classes.append(images_class_2)
    images_classes.append(images_class_3)
    images_classes.append(images_class_4)

    nb_classes = 5
    for idx_cl in range(nb_classes):
        images = images_classes[idx_cl][:nb_imgs, :, :].reshape(nb_imgs, x_train.shape[1], x_train.shape[2], 1)
        plot_att_response_maps(images, model_arch_file=model_arch_file, model_weights_file=model_weights_file,
                               cmap=cm.viridis,
                               figure_dir=figure_dir+'_class'+str(idx_cl), nb_conv_layers=6, nb_top_feat_maps=4, layer_nb='all',
                               plot_all_filters=True, plot_filter_sum=True, plot_summary=True)
