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
    import os

    # os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["KERAS_BACKEND"] = "theano"

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
    from sklearn.metrics import accuracy_score
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
    from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps

    startTime = datetime.now()
    now = datetime.now()

    # read config file
    config_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # directories
    main_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/'
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
    checkpoint_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets_2d')))
    figure_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))

    # files
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))


    ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_7_classes_new.db'

    ase_atoms_list = read_ase_db(db_path=ase_db_file)

    images = []
    for ase_atoms in ase_atoms_list:
        images.append(ase_atoms.info['descriptor']['diffraction_2d_intensity'])

    neural_network_name = 'ziletti_et_2018_rgb'
    model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_folder, neural_network_name + '.h5')))
    model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_folder, neural_network_name + '.json')))

    images = np.asarray(images)
    logger.info("images.shape: {}".format(images.shape))

    plot_att_response_maps(images, model_arch_file, model_weights_file, figure_folder, nb_conv_layers=6, nb_top_feat_maps=4,
                           layer_nb='all', plot_all_filters=True, plot_filter_sum=True, plot_summary=True)

    sys.exit(1)
