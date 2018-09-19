#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016-2018, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/04/18"

if __name__ == "__main__":

    import sys
    import os.path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    common_dir = os.path.normpath(os.path.join(base_dir, "../../../python-common/common/python"))
    nomadml_dir = os.path.normpath(os.path.join(base_dir, "../python-modules/"))
    atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data'))
    apt_dir = os.path.normpath(os.path.join(base_dir, "../../../apt/"))

    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
        sys.path.insert(0, nomadml_dir)
        sys.path.insert(0, atomic_data_dir)
        sys.path.insert(0, apt_dir)

    from datetime import datetime
    from functools import partial
    import numpy as np
    import math
    from scipy import stats
    from sklearn.metrics import accuracy_score
    import six.moves.cPickle as pickle

    from sklearn.metrics import confusion_matrix

    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import make_data_sets
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.models.cnn_nature_comm_ziletti2018 import model_deep_cnn_struct_recognition
    from ai4materials.models.cnn_nature_comm_ziletti2018 import train_cnn_keras
    from ai4materials.models.cnn_nature_comm_ziletti2018 import predict_cnn_keras
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import get_spacegroup
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.utils.utils_plotting import plot_confusion_matrix
    from ai4materials.wrappers import load_descriptor

    startTime = datetime.now()
    now = datetime.now()

    # read config file
    config_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # directories
    main_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/'
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets_2d')))

    # files
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))

    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_crossover.csv')))


    # =============================================================================
    # Load Descriptor file and Dataset preparation
    # =============================================================================

    spacegroups = ['139', '141', '166', '194', '221', '225', '227', '229']
    input_dims = (64, 64)

    new_labels = {"bct_139": ["139"], "bct_141": ["141"], "hex/rh": ["166", "194"],
                  "sc": ["221"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]}

    # spacegroups = ['166', '221']

    # =============================================================================
    # Prepare Pristine Dataset (training set)
    # =============================================================================

    # desc_file_path = []

    # for spacegroup in spacegroups:
    #     desc_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/spgroup_' + str(spacegroup) + \
    #                 '_pristine.tar.gz'
    #     desc_file_path.append(desc_file)
    #
    # target_list, ase_atoms_list_pristine = load_descriptor(desc_files=desc_file_path, configs=configs)
    #
    # ase_db_file_pristine = write_ase_db(ase_atoms_list=ase_atoms_list_pristine,
    #                                     db_name='elemental_solids_ncomms_1e-3_1e-6_pristine', main_folder=main_folder,
    #                                     folder_name='db_ase')
    #
    # path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(
    #     structure_list=ase_atoms_list_pristine,
    #     target_list=target_list,
    #     desc_metadata='diffraction_2d_intensity',
    #     dataset_name='pristine_dataset',
    #     target_name='spacegroup_nb_symprec_1e-06',
    #     target_categorical=True,
    #     input_dims=input_dims,
    #     configs=configs,
    #     dataset_folder=dataset_folder,
    #     main_folder=main_folder,
    #     desc_folder=desc_folder,
    #     tmp_folder=tmp_folder,
    #     disc_type=None, n_bins=None,
    #     notes="Spglib thresholds are 1e-3, 1e-6, 1e-9 for all apart 166 and 194, for which we use 1e-3, 1e-6.",
    #     new_labels=new_labels)
    #

    # =============================================================================
    # Prepare Defective Datasets (test set)
    # =============================================================================

    vacs = ['0.01', '0.02', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7']

    # for vac in vacs:
    #     desc_file_path = []
    #     logger.info("Vac ratio: {}".format(vac))
    #     for spacegroup in spacegroups:
    #         desc_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/spgroup_' + str(spacegroup) + \
    #                     '_vac' + str(vac) + '.tar.gz'
    #
    #         desc_file_path.append(desc_file)
    #
    #     target_list, ase_atoms_list = load_descriptor(desc_files=desc_file_path, configs=configs)
    #
    #     path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(
    #         structure_list=ase_atoms_list,
    #         target_list=target_list,
    #         desc_metadata='diffraction_2d_intensity',
    #         dataset_name='vac' + str(vac) + '_dataset',
    #         target_name='spacegroup_nb_symprec_1e-06',
    #         target_categorical=True,
    #         input_dims=input_dims,
    #         configs=configs,
    #         dataset_folder=dataset_folder,
    #         main_folder=main_folder,
    #         desc_folder=desc_folder,
    #         tmp_folder=tmp_folder,
    #         disc_type=None, n_bins=None,
    #         notes="Spglib thresholds are 1e-3, 1e-6, 1e-9 for all apart 166-194 for which we use 1e-3, 1e-6.",
    #         new_labels=new_labels)

        # ase_db_file = write_ase_db(ase_atoms_list=ase_atoms_list,
        #                            db_name='elemental_solids_ncomms_vac' + str(vac), main_folder=main_folder,
        #                            folder_name='db_ase')

    disps = ['0.001', '0.002', '0.003', '0.004', '0.005', '0.01', '0.02', '0.04', '0.06', '0.08', '0.1']

    # for disp in disps:
    #     desc_file_path = []
    #     logger.info("Disp: {}".format(disp))
    #     for spacegroup in spacegroups:
    #         desc_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/spgroup_' + str(spacegroup) + \
    #                     '_disp' + str(disp) + '.tar.gz'
    #
    #         desc_file_path.append(desc_file)
    #
    #     target_list, ase_atoms_list = load_descriptor(desc_files=desc_file_path, configs=configs)
    #
    #     path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(
    #         structure_list=ase_atoms_list,
    #         target_list=target_list,
    #         desc_metadata='diffraction_2d_intensity',
    #         dataset_name='disp' + str(disp) + '_dataset',
    #         target_name='spacegroup_nb_symprec_1e-06',
    #         target_categorical=True,
    #         input_dims=input_dims,
    #         configs=configs,
    #         dataset_folder=dataset_folder,
    #         main_folder=main_folder,
    #         desc_folder=desc_folder,
    #         tmp_folder=tmp_folder,
    #         disc_type=None, n_bins=None,
    #         notes="Spglib thresholds are 1e-3, 1e-6, 1e-9 for all apart 166-194 for which we use 1e-3, 1e-6.",
    #         new_labels=new_labels)


    #     ase_db_file = write_ase_db(ase_atoms_list=ase_atoms_list,
    #                                db_name='elemental_solids_ncomms_disp' + str(disp), main_folder=main_folder,
    #                                folder_name='db_ase')

    # =============================================================================
    # Read Training and Test Datasets
    # =============================================================================

    train_set_name = 'pristine_dataset'
    path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(dataset_folder, train_set_name + '_summary.json')))

    test_set_name = 'disp0.1_dataset'
    path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_x.pkl')))
    path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_y.pkl')))
    path_to_summary_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_summary.json')))


    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                               path_to_summary=path_to_summary_test)


    params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
                  "classes": dataset_info_train["data"][0]["classes"],
                  # "checkpoint_filename": 'try_'+str(now.isoformat()),
                  "checkpoint_filename": 'ziletti_et_2018_rgb',
                  "batch_size": 32, "img_channels": 3}

    text_labels = np.asarray(dataset_info_train["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_train["data"][0]["numerical_labels"])
    classes = dataset_info_train["data"][0]["classes"]

    # text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    # numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

    data_set = make_data_sets(x_train_val=x_train, y_train_val=y_train,
                              x_test=x_test, y_test=y_test,
                              split_train_val=True, test_size=0.1,
                              stratified_splits=True)

    # =============================================================================
    # Neural network training and prediction
    # =============================================================================

    partial_model_architecture = partial(model_deep_cnn_struct_recognition, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                         kernel_sizes=[7, 7, 7, 7, 7, 7], max_pool_strides=[2, 2], hidden_layer_size=128)

    # generate image of architecture
    # train_cnn_keras(
    #    data_set=data_set,
    #    configs=configs, batch_size=params_cnn["batch_size"],
    #    nb_classes=params_cnn["nb_classes"], img_channels=3,
    #    data_augmentation=False,
    #    partial_model_architecture=partial_model_architecture,
    #    normalize=True,
    #    checkpoint_dir=checkpoint_dir,
    #    checkpoint_filename=params_cnn["checkpoint_filename"],
    #    nb_epoch=5,
    #    training_log_file=training_log_file,
    #    early_stopping=False)

    target_pred_class, target_pred_probs, prob_predictions, conf_matrix = predict_cnn_keras(
        data_set=data_set,
        nb_classes=params_cnn["nb_classes"],
        configs=configs,
        batch_size=params_cnn["batch_size"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=params_cnn["checkpoint_filename"],
        show_model_acc=True,
        predict_probabilities=True,
        plot_conf_matrix=True,
        conf_matrix_file=conf_matrix_file,
        numerical_labels=numerical_labels,
        text_labels=text_labels,
        results_file=results_file)

    sys.exit(1)

    # y_nn = [classes[item] for item in target_pred_class]

    # ase_db_file_pristine = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_pristine.db'

    ase_db_file_pristine = [
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_139.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_141.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_166.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_194.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_221.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_225.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_227.db',
        '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_229.db'
    ]

        # ase_db_file_vac20 = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_vac0.2.db'
    # ase_db_file_pristine = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_7_classes.db'
    # ase_db_file_vac20 = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_vac0.2.db'
    # ase_atoms_list_vac20 = read_ase_db(db_path=ase_dpb_file_vac20)

    ase_atoms_list_pristine = []
    for db_path in ase_db_file_pristine:
        ase_atoms_list_pristine.extend(read_ase_db(db_path=db_path))

    symprec = 1.e-9
    angle_tolerance = -1.0
    y_true = []
    for idx, ase_atom_pristine in enumerate(ase_atoms_list_pristine):
        if idx % (int(len(ase_atoms_list_pristine) / 10) + 1) == 0:
            logger.debug("Reading ASE atom structure: file {0}/{1}".format(idx + 1, len(ase_atoms_list_pristine)))
        y_true.extend(get_spacegroup(ase_atom_pristine, symprec=symprec, angle_tolerance=angle_tolerance))

    true_labels_filename = 'y_true_list' + '_symprec' + str(symprec) + '_angletol' + str(angle_tolerance)
    true_labels_filename_path = os.path.abspath(os.path.normpath(os.path.join(main_folder, true_labels_filename + '.pkl')))

    with open(true_labels_filename_path, 'wb') as f:
        pickle.dump(y_true, f)




    with open(true_labels_filename_path, 'rb') as f:
        y_true = pickle.load(f)
        logger.info(stats.itemfreq(y_true))

    sys.exit()


    with open(true_labels_filename_path, 'rb') as f:
        y_true = pickle.load(f)

    logger.info("Y_true: {}".format(y_true))

    sys.exit()
    # y_spglib = []
    # symprec = 0.001
    # for ase_atom_vac20 in ase_atoms_list_vac20:
    #     y_spglib.extend(get_spacegroup(ase_atom_vac20, symprec=symprec))
    # y_spglib = [str(item) for item in y_spglib]

    # redefine labels
    if new_labels is not None:
        # y_spglib = [str(item) for item in y_spglib]
        y_true = [str(item) for item in y_true]

        for key in new_labels.keys():
            # y_spglib = [key if item in new_labels[key] else item for item in y_spglib]
            # y_spglib = np.asarray(y_spglib)

            y_true = [key if item in new_labels[key] else item for item in y_true]
            y_true = np.asarray(y_true)

    logger.info("Accuracy score Spglib: {}".format(accuracy_score(y_true, y_spglib)))
    logger.info("Accuracy score Neural Network: {}".format(accuracy_score(y_true, y_nn)))

    sys.exit(1)
