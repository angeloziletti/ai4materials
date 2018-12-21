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

    from ase.io.trajectory import Trajectory
    from ase.spacegroup import crystal
    from ai4materials.descriptors.diffraction3d import DISH
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_plotting import aggregate_struct_trans_data
    from ai4materials.utils.utils_plotting import make_crossover_plot
    from ai4materials.utils.utils_crystals import get_md_structures
    from ai4materials.wrappers import load_descriptor
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.utils.utils_data_retrieval import generate_facets_input
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.wrappers import calc_descriptor_in_memory
    from argparse import ArgumentParser
    from datetime import datetime
    import numpy as np
    from keras.models import load_model
    from ai4materials.models.cnn_polycrystals import predict
    from mendeleev import element
    from ai4materials.utils.utils_data_retrieval import read_ase_db

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
        db_files_prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase/'

    # read config file
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_melting_cu.csv')))
    filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))

    configs['io']['dataset_folder'] = dataset_folder
    configs['io']['desc_folder'] = desc_folder

    descriptor = DISH(configs=configs)

    # =============================================================================
    # Read prototype data from files
    # =============================================================================

    steps_t = 61
    nb_samples = 20
    min_target_t = 0.
    max_target_t = 600.

    target_temps = np.linspace(min_target_t, max_target_t, steps_t)

    ase_atoms_list = get_md_structures(min_target_t=min_target_t, max_target_t=max_target_t, steps_t=steps_t,
                                       nb_samples=nb_samples,
                                       max_nb_trials=100000, backend='asap', supercell_size=4)

    ase_db_filename = write_ase_db(ase_atoms_list, main_folder, db_name='cu_copper', db_type='db', overwrite=True,
                         folder_name='db_ase')  # for item in ase_atoms_list:

    ase_atoms_list = read_ase_db(ase_db_filename)

    desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                               tmp_folder=configs['io']['tmp_folder'],
                                               desc_folder=configs['io']['desc_folder'],
                                               desc_file='melting_cu_55.tar.gz', format_geometry='aims',
                                               operations_on_structure=None,
                                               nb_jobs=6)

    desc_file_path = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/melting_cu_55.tar.gz'

    # now prepare the dataset
    # load the previously saved file containing the crystal structures and their corresponding descriptor
    target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    # sort the structures according to the original label
    structure_list.sort(key=lambda x: int(x.info['label'].split('struct-')[1]))

    path_to_x, path_to_y, path_to_summary = prepare_dataset(structure_list=structure_list, target_list=target_list,
                                                            desc_metadata='diffraction_3d_sh_spectrum',
                                                            dataset_name='cu_melting', target_name='target',
                                                            target_categorical=True, input_dims=(52, 32),
                                                            configs=configs,
                                                            dataset_folder=configs['io']['dataset_folder'],
                                                            main_folder=configs['io']['main_folder'],
                                                            desc_folder=configs['io']['desc_folder'],
                                                            tmp_folder=configs['io']['tmp_folder'])

    train_set_name = 'hcp-sc-fcc-diam-bcc_pristine'
    path_to_x_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')))

    test_set_name = 'cu_melting'

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

    params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"], "batch_size": 32}

    text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

    # load trained neural network from hdf5 file
    path_to_saved_model = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models/enc_dec_drop12.5/model.h5'
    model = load_model(path_to_saved_model)

    conf_matrix_file = os.path.abspath(
        os.path.normpath(os.path.join(main_folder, 'confusion_matrix_' + test_set_name + '.png')))

    results = predict(x=x_test, y=y_test, configs=configs, numerical_labels=numerical_labels, text_labels=text_labels,
                      nb_classes=params_cnn["nb_classes"], model=model, batch_size=params_cnn["batch_size"],
                      conf_matrix_file=conf_matrix_file, results_file=results_file, mc_samples=1000)

    df_results = aggregate_struct_trans_data(results_file, nb_rows_to_cut=0, nb_samples=nb_samples,
                                             nb_order_param_steps=steps_t,
                                             min_order_param=min_target_t,
                                             max_order_param=max_target_t, prob_idxs=[0, 1, 2, 3, 4])

    make_crossover_plot(df_results, results_file, prob_idxs=[0, 1, 2, 3, 4],
                        labels=["$p_{hcp}$", "$p_{sc}$", "$p_{fcc}$", "$p_{diam}$", "$p_{bcc}$"],
                        nb_order_param_steps=steps_t, filename_suffix=".png", title="Rocksalt with different Delta Z",
                        x_label="Delta Z", show_plot=False, markersize=3.0)

    make_crossover_plot(df_results, results_file, plot_type='uncertainty', labels=["$mc_{1000}$"],
                        nb_order_param_steps=steps_t, filename_suffix=".svg", title="From rocksalt to fcc",
                        x_label="Central atoms removed (%)", show_plot=False, markersize=1.0, palette=['black'])
