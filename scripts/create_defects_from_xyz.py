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
__copyright__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/04/18"

if __name__ == "__main__":
    import sys
    import os.path

    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from datetime import datetime
    import numpy as np
    import webbrowser
    import ase
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_crystals import create_vacancies

    startTime = datetime.now()
    now = datetime.now()

    main_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/'
    prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/prototypes_aflow_new'
    db_files_prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase_prototypes'

    # read config file
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models/enc_dec')))
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
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

    structure_files = []
    structure_files.append(os.path.join(main_folder, 'structures_for_paper/edge_dislocation/small_edge_dislocation.xyz'))

    for structure_file in structure_files:
        atoms = ase.io.read(structure_file, index=0, format='xyz')

        # atoms_mod = random_displace_atoms(atoms, displacement_scaled=0.01, create_replicas_by='user-defined',
        #                                   cell_type=None, target_replicas=[1, 1, 1],
        #                                   noise_distribution='uniform_scaled', target_nb_atoms=128,
        #                                   random_rotation=False, optimal_supercell=False)

        atoms_mod2 = create_vacancies(atoms, target_vacancy_ratio=0.05, create_replicas_by='user-defined',
                                          cell_type=None, target_replicas=[1, 1, 1],
                                          random_rotation=False, optimal_supercell=False)

        structure_name, file_extension = os.path.splitext(structure_file)

        ase.io.write(structure_name + '_vac5.xyz', atoms_mod2, format='xyz', parallel=True)

