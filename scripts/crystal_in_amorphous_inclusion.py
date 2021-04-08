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
    logger = setup_logger(configs, level='DEBUG', display_configs=False)

    # setup folder and files
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file_cu = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_melting_cu.csv')))
    results_file_fe = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_melting_fe.csv')))

    filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))

    configs['io']['dataset_folder'] = dataset_folder
    configs['io']['desc_folder'] = desc_folder

    descriptor = DISH(configs=configs)

    # =============================================================================
    # Read prototype data from files
    # =============================================================================


    # Fe (BCC) - from BCC to amorphous
    steps_t = 1
    nb_samples = 1
    min_target_t = 2000.
    max_target_t = 2000.

    ase_atoms_list = get_md_structures(min_target_t=min_target_t, max_target_t=max_target_t, steps_t=steps_t,
                                       nb_samples=nb_samples, element='Fe', backend='asap',
                                       max_nb_trials=100000, supercell_size=10)

    # ase_db_filename = write_ase_db(ase_atoms_list, main_folder, db_name='try1',
    #     db_type = 'db', overwrite = True, folder_name = 'db_ase')

    desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                               tmp_folder=configs['io']['tmp_folder'],
                                               desc_folder=configs['io']['desc_folder'],
                                               desc_file='amorphous_bcc_try1.tar.gz',
                                               format_geometry='aims',
                                               operations_on_structure=None,
                                               nb_jobs=1)

    # desc_file_path = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/cu_copper_61steps_0_600_sc4.tar.gz'
    # desc_file_path = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/fe_21steps_0_2000_sc4.tar.gz'