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

    from functools import partial
    import numpy as np
    import math
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import make_data_sets
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.descriptors.diffraction2d import Diffraction2D
    from ai4materials.models.cnn_nature_comm_ziletti2018 import model_deep_cnn_struct_recognition
    from ai4materials.models.cnn_nature_comm_ziletti2018 import train_cnn_keras
    from ai4materials.models.cnn_nature_comm_ziletti2018 import predict_cnn_keras
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_crystals import spacegroup_a_to_spacegroup_b
    from ai4materials.utils.utils_data_retrieval import generate_facets_input
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.utils.utils_parsing import read_data
    from ai4materials.utils.utils_plotting import aggregate_struct_trans_data
    from ai4materials.utils.utils_plotting import make_crossover_plot
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.wrappers import get_json_list
    from ai4materials.wrappers import load_descriptor


    # read config file
    config_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # directories
    main_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/'
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'bcc_to_sc')))
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets_2d')))

    # files
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_crossover.csv')))
    # =============================================================================
    #  Define descriptor
    # =============================================================================

    user_param_source = {'wavelength': 5.0E-12, 'pulse_energy': 1E-6, 'focus_diameter': 1E-6}

    user_param_detector = {'distance': 0.1, 'pixel_size': 4E-4, 'nx': 64, 'ny': 64}


    # =============================================================================
    # Rotation matrix for each channel
    # =============================================================================

    def rot_mat_x(angle):
        return np.array([[1, 0, 0], [0, math.cos(np.radians(angle)), math.sin(np.radians(angle))],
                         [0, -math.sin(np.radians(angle)), math.cos(np.radians(angle))]]).astype(float)


    def rot_mat_y(angle):
        return np.array([[math.cos(np.radians(angle)), 0, math.sin(np.radians(angle))], [0, 1, 0],
                         [-math.sin(np.radians(angle)), 0, math.cos(np.radians(angle))]]).astype(float)


    def rot_mat_z(angle):
        return np.array([[math.cos(np.radians(angle)), math.sin(np.radians(angle)), 0],
                         [-math.sin(np.radians(angle)), math.cos(np.radians(angle)), 0], [0, 0, 1]]).astype(float)


    desc_angles = {"r": [-45., 45.], "g": [-45., 45.], "b": [-45., 45.]}

    rot_matrices = {}
    rot_matrices_x = []
    for desc_angle in desc_angles["r"]:
        rot_matrices_x.append(rot_mat_x(desc_angle))
    rot_matrices["r"] = rot_matrices_x

    rot_matrices_y = []
    for desc_angle in desc_angles["g"]:
        rot_matrices_y.append(rot_mat_y(desc_angle))
    rot_matrices["g"] = rot_matrices_y

    rot_matrices_z = []
    for desc_angle in desc_angles["b"]:
        rot_matrices_z.append(rot_mat_z(desc_angle))
    rot_matrices["b"] = rot_matrices_z

    input_dims = (64, 64)

    kwargs = dict(mask_r_min=5, user_param_source=user_param_source, user_param_detector=user_param_detector,
                  atoms_scaling='avg_nn', use_mask=True, rot_matrices=rot_matrices,
                  atoms_scaling_cutoffs=[4.0, 5.0, 7.0, 9.0, 11.0, 12.0])

    descriptor = Diffraction2D(configs=configs, **kwargs)

    # ==========================================================================================
    # Write ASE db files - one sample for each one of the 7 classes - used later in prediction
    # ==========================================================================================


    # desc_file_7_classes_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
    #                                            tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
    #                                            desc_file='7_classes', format_geometry='aims',
    #                                            operations_on_structure=operations_on_structure_list, nb_jobs=1, **kwargs)
    #
    # target_list, structure_list = load_descriptor(desc_files=desc_file_7_classes_path, configs=configs)
    #
    # df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_2d_intensity',
    #                                          target_list=target_list, sprite_atlas_filename='desc_7_classes', configs=configs,
    #                                          normalize=True)
    #
    # logger.info("Descriptor calculation completed.")
    #
    #
    # ase_db_file_7_classes = write_ase_db(ase_atoms_list=ase_atoms_list,
    #                                      db_name='elemental_solids_ncomms_7_classes', main_folder=main_folder,
    #                                      folder_name='db_ase')


    # ==============================================================================
    # Write ASE db files - prototypes for bcc-->sc transition
    # ==============================================================================

    bcc_json_list = []
    prefix_file = "/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_cI2_229_a/"
    suffix_file = "A_cI2_229_a_4.0_B_nomad.json"
    for i in range(0, 1):
        bcc_json_list.append(prefix_file+suffix_file)

    ase_atoms_list = read_data(bcc_json_list, calc_spgroup=True, symprec=[1e-03, 1e-06, 1e-09])
    ase_db_file = write_ase_db(ase_atoms_list=ase_atoms_list, db_name='bcc_prototype', main_folder=main_folder,
                               folder_name='db_ase')

    # =============================================================================
    # Descriptor calculation - bcc-->sc
    # =============================================================================

    # desc_file_name = 'structural_transition_bcc_sc'
    # ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/bcc_prototype.db'
    # ase_atoms_list = read_ase_db(db_path=ase_db_file)
    #
    # target_b_contributions = np.linspace(0.0, 1.0, num=21).tolist()
    #
    # for target_b_contrib in target_b_contributions:
    #     # define operations on structures
    #     operations_on_structure_list = [[spacegroup_a_to_spacegroup_b,
    #                                      dict(spgroup_a='Im-3m', spgroup_b='Pm-3m', target_b_contribution=target_b_contrib,
    #                                           cell_type='standard_no_symmetries',
    #                                           create_replicas_by='user-defined', min_nb_atoms=32, target_nb_atoms=256,
    #                                           target_replicas=[6, 6, 6])]]
    #
    #     desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
    #                                      tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
    #                                      desc_file=str(target_b_contrib) + '_bcc_to_sc.json.tar.gz', format_geometry='aims',
    #                                      operations_on_structure=operations_on_structure_list*80, nb_jobs=-1, **kwargs)
    #
    # bcc_to_sc_list = []
    # prefix_file = "/home/ziletti/Documents/calc_xray/2d_nature_comm/bcc_to_sc/"
    # suffix_file = "_bcc_to_sc.json.tar.gz"
    #
    # for target_b_contrib in target_b_contributions:
    #     bcc_to_sc_list.append(prefix_file+str(target_b_contrib)+suffix_file)
    #
    # target_list, structure_list = load_descriptor(desc_files=bcc_to_sc_list, configs=configs)
    #
    # df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_2d_intensity',
    #                                          target_list=target_list, sprite_atlas_filename=desc_file_name, configs=configs,
    #                                          normalize=True)
    #
    # logger.info("Descriptor calculation completed.")

    # =============================================================================
    # Prepare Structural bcc --> sc transition Dataset
    # =============================================================================
    prefix_file = "/home/ziletti/Documents/calc_xray/2d_nature_comm/bcc_to_sc/"
    suffix_file = "_bcc_to_sc.json.tar.gz"
    target_b_contributions = np.linspace(0.0, 1.0, num=21).tolist()
    bcc_to_sc_list = []
    for target_b_contrib in target_b_contributions:
        bcc_to_sc_list.append(prefix_file + str(target_b_contrib) + suffix_file)

    desc_file_path = []
    desc_file_7_classes = ['/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/7_classes.tar.gz']
    desc_file_path.extend(desc_file_7_classes)
    desc_file_path.extend(bcc_to_sc_list)

    target_list, ase_atoms_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    new_labels = {"bct_139": ["139"], "bct_141": ["141"], "hex/rh": ["166", "194"],
                  "sc": ["221"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]}

    path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(structure_list=ase_atoms_list,
                                                                              target_list=target_list,
                                                                              desc_metadata='diffraction_2d_intensity',
                                                                              dataset_name='bcc_to_sc',
                                                                              target_name='spacegroup_nb_symprec_1e-06',
                                                                              target_categorical=True,
                                                                              input_dims=input_dims, configs=configs,
                                                                              dataset_folder=dataset_folder,
                                                                              main_folder=main_folder,
                                                                              desc_folder=desc_folder,
                                                                              tmp_folder=tmp_folder, disc_type=None,
                                                                              n_bins=None,
                                                                              notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'. Added 7 examples (one for each class)",
                                                                              new_labels=new_labels)

    train_set_name = 'pristine_dataset'
    path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(dataset_folder, train_set_name + '_summary.json')))

    test_set_name = 'bcc_to_sc'
    path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_x.pkl')))
    path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_y.pkl')))
    path_to_summary_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_summary.json')))

    x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                               path_to_summary=path_to_summary_test)

    params_cnn = {"nb_classes": dataset_info_test["data"][0]["nb_classes"],
                  "classes": dataset_info_test["data"][0]["classes"], "checkpoint_filename": 'ziletti_et_2018_rgb',
                  "batch_size": 32, "img_channels": 3}

    text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])
    classes = dataset_info_test["data"][0]["classes"]
    #
    data_set = make_data_sets(x_train_val=x_test, y_train_val=y_test, x_test=x_test, y_test=y_test, split_train_val=False,
                              test_size=0.1, stratified_splits=False)

    # # =============================================================================
    # # Neural network prediction
    # # =============================================================================
    #
    partial_model_architecture = partial(model_deep_cnn_struct_recognition, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                         kernel_sizes=[7, 7, 7, 7, 7, 7], max_pool_strides=[2, 2], hidden_layer_size=128)

    target_pred_class, target_pred_probs, prob_predictions, conf_matrix = predict_cnn_keras(data_set=data_set,
                                                                                            nb_classes=params_cnn[
                                                                                                "nb_classes"],
                                                                                            configs=configs,
                                                                                            batch_size=params_cnn[
                                                                                                "batch_size"],
                                                                                            checkpoint_dir=checkpoint_dir,
                                                                                            checkpoint_filename=params_cnn[
                                                                                                "checkpoint_filename"],
                                                                                            show_model_acc=True,
                                                                                            predict_probabilities=True,
                                                                                            plot_conf_matrix=True,
                                                                                            conf_matrix_file=conf_matrix_file,
                                                                                            numerical_labels=numerical_labels,
                                                                                            text_labels=text_labels,
                                                                                            results_file=results_file)

    # ==============================================================================
    # Plot Structural transition - bcc-->rh-->sc-->rh-->fcc-->sc
    # ==============================================================================

    palette = ['indigo', 'saddlebrown', 'black', 'green', 'blue', 'gold', 'red']
    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"]

    df_results = aggregate_struct_trans_data(results_file, nb_rows_to_cut=7, nb_samples=80, nb_order_param_steps=21,
                                             min_order_param=0.0, max_order_param=1.0,
                                             prob_idxs=range(params_cnn["nb_classes"]))

    make_crossover_plot(df_results, results_file, prob_idxs=[0, 6], palette=palette, labels=labels,
                        linewidth=2.0, markersize=4.0,
                        nb_order_param_steps=21, max_nb_ticks=21, filename_suffix=".svg",
                        title="Structural transition: bcc->sc", x_label="q", show_plot=False)
