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
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_plotting import aggregate_struct_trans_data
    from ai4materials.utils.utils_plotting import make_crossover_plot
    from ai4materials.wrappers import load_descriptor
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
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))

    configs['io']['dataset_folder'] = dataset_folder
    configs['io']['desc_folder'] = desc_folder

    descriptor = DISH(configs=configs)
    # descriptor = Diffraction2D(configs=configs)

    target_nb_atoms = 128
    nb_rotations = 5

    nb_order_param_steps = 104
    min_order_param = 0.4
    max_order_param = 0.5
    # define operations on structures
    target_vacancy_ratios = np.linspace(min_order_param, max_order_param, nb_order_param_steps, endpoint=True)

    nb_samples = 20
    a = 5.0
    nacl_structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225, cellpar=[a, a, a, 90, 90, 90])

    # ase_atoms_list = []
    # for target_vacancy_ratio in target_vacancy_ratios:
    #
    #     for sample in range(nb_samples):
    #         nacl_structure_def = create_vacancies(nacl_structure, target_vacancy_ratio=target_vacancy_ratio,
    #                                               target_species='Cl', create_replicas_by='user-defined',
    #                                               target_replicas=(3, 3, 3), random_rotation_before=True,
    #                                               cell_type='standard_no_symmetries', optimal_supercell=False)
    #
    #         ase_atoms_list.append(nacl_structure_def)

    # desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
    #                                            tmp_folder=configs['io']['tmp_folder'],
    #                                            desc_folder=configs['io']['desc_folder'],
    #                                            desc_file='rocksalt_to_fcc_104_20samples.tar.gz', format_geometry='aims',
    #                                            # operations_on_structure=operations_on_structure_list[0],
    #                                            operations_on_structure=None,
    #                                            nb_jobs=6)  # operations_on_structure=None, nb_jobs=1)

    desc_file_path = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/rocksalt_to_fcc_104_20samples.tar.gz'
    # desc_file_path = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/rocksalt_to_fcc.tar.gz'

    # now prepare the dataset
    # load the previously saved file containing the crystal structures and their corresponding descriptor
    # target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    # sort the structures according to the original label
    # structure_list.sort(key=lambda x: int(x.info['label'].split('struct-')[1]))

    # create a texture atlas with all the two-dimensional diffraction fingerprints
    # df, texture_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_3d_sh_spectrum',
    #                                           target_list=target_list, sprite_atlas_filename=desc_file_path,
    #                                           configs=configs, normalize=True)

    # path_to_x, path_to_y, path_to_summary = prepare_dataset(structure_list=structure_list, target_list=target_list,
    #                                                         desc_metadata='diffraction_3d_sh_spectrum',
    #                                                         dataset_name='rocksalt_to_fcc', target_name='target',
    #                                                         target_categorical=True, input_dims=(52, 32),
    #                                                         configs=configs,
    #                                                         dataset_folder=configs['io']['dataset_folder'],
    #                                                         main_folder=configs['io']['main_folder'],
    #                                                         desc_folder=configs['io']['desc_folder'],
    #                                                         tmp_folder=configs['io']['tmp_folder'])

    train_set_name = 'hcp-sc-fcc-diam-bcc_pristine'
    path_to_x_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')))

    test_set_name = 'rocksalt_to_fcc'

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
                      conf_matrix_file=conf_matrix_file, results_file=results_file, mc_samples=2)

    df_results = aggregate_struct_trans_data(results_file, nb_rows_to_cut=0, nb_samples=nb_samples,
                                             nb_order_param_steps=nb_order_param_steps,
                                             min_order_param=min_order_param,
                                             max_order_param=max_order_param,
                                             prob_idxs=[1, 2],
                                             with_uncertainty=True)

    make_crossover_plot(df_results, results_file, prob_idxs=[1, 2],
                        labels=["$p_{hcp}$", "$p_{sc}$", "$p_{fcc}$", "$p_{diam}$", "$p_{bcc}$"],
                        nb_order_param_steps=21, filename_suffix=".svg", title="From rocksalt to fcc",
                        x_label="Central atoms removed (%)", show_plot=False, markersize=1.0,
                        palette=['yellow', 'red', 'blue', 'green', 'indigo'])

    import pandas as pd

    df = pd.read_csv(results_file)

    sample_avg_mut_info = []
    sample_avg_pred_entropy = []
    # use n_samples to aggregate results
    for idx_row in range(0, len(df), nb_samples):
        # extract nb_samples consecutive samples and average them
        # then move down nb_samples and continue till the end
        sample_avg_mut_info.append(
            df.iloc[idx_row:idx_row + nb_samples]['uncertainty_mutual_information'].values.mean())
        sample_avg_pred_entropy.append(
            df.iloc[idx_row:idx_row + nb_samples]['uncertainty_predictive_entropy'].values.mean())

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(sample_avg_mut_info)
    plt.savefig(os.path.join(main_folder, 'mutual_info.png'))

    plt.clf()
    plt.plot(sample_avg_pred_entropy)
    plt.savefig(os.path.join(main_folder, 'pred_entropy.png'))
