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



    from ase.spacegroup import get_spacegroup
    from ai4materials.descriptors.diffraction3d import Diffraction3D
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.models.strided_pattern_matching import get_classification_map
    from ai4materials.visualization.viewer import Viewer
    from ai4materials.utils.utils_data_retrieval import generate_facets_input
    from ai4materials.utils.utils_parsing import read_atomic_structures
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import make_data_sets
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.visualization.viewer import Viewer
    from ai4materials.models.strided_pattern_matching import get_structures_by_boxes
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.wrappers import load_descriptor
    import numpy as np
    from ai4materials.models.cnn_architectures import model_cnn_rot_inv, model_fully_conv
    from argparse import ArgumentParser
    from functools import partial
    from datetime import datetime
    import numpy as np
    import webbrowser

    startTime = datetime.now()
    now = datetime.now()

    parser = ArgumentParser()
    parser.add_argument("-m", "--machine", dest="machine", help="on which machine the script is run", metavar="MACHINE")
    args = parser.parse_args()

    machine = vars(args)['machine']
    # machine = 'eos'

    if machine == 'eos':
        config_file = '/scratch/ziang/polycrystals/config_prototypes.yml'
        main_folder = '/scratch/ziang/polycrystals/'
        prototypes_basedir = '/scratch/ziang/polycrystals/prototypes_aflow_new/'
        db_files_prototypes_basedir = '/scratch/ziang/polycrystals/db_ase_prototypes'

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
    structure_files.append(os.path.join(main_folder, 'structures_for_paper/grain_boundaries/0012262150_v6bxv2_tv0.4bxv0.3_d2.1z_traj.xyz'))
    structure_files.append(os.path.join(main_folder, 'structures_for_paper/inclusions/Fe_bcc_Si_fcc_final.xyz'))

    descriptor = Diffraction3D(configs=configs)

    operations_on_structure_list = [
        (create_supercell, dict(create_replicas_by='user-defined', target_replicas=[1, 1, 1], random_rotation=False)), (
            create_vacancies, dict(target_vacancy_ratio=0.20, create_replicas_by='user-defined', cell_type=None,
                                   target_replicas=[1, 1, 1], random_rotation=False, optimal_supercell=False)), (
        random_displace_atoms,
        dict(displacement_scaled=0.01, create_replicas_by='user-defined', cell_type=None, target_replicas=[1, 1, 1],
             noise_distribution='uniform_scaled', target_nb_atoms=128, random_rotation=False, optimal_supercell=False))]

    # =============================================================================
    # Descriptor calculation
    # =============================================================================
    # stride_size = [0.5, 0.5, 20.0]
    stride_size = [20.0, 20.0, 20.0]
    box_sizes = [10.0, 10.0, 10.0, 10.0]
    padding_ratios = [(0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    # desc_file = None

    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac50.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac70.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac80.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac90.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')

    desc_file = os.path.join(main_folder, 'desc_folder/grain_boundaries/0012262150_v6bxv2_tv0.4bxv0.3_d2.1z_traj.xyz_stride_1.0_1.0_1.0_box_size_10.0_.tar.gz')

    for idx, structure_file in enumerate(structure_files):
        get_classification_map(structure_file, descriptor, 'diffraction_3d_sh_spectrum', configs,
                               desc_only=False,
                               operations_on_structure=operations_on_structure_list[0], stride_size=stride_size,
                               box_size=box_sizes[idx],
                               # box_size=None,
                               # init_sliding_volume=(16., 16., 16.),
                               desc_file=desc_file, show_plot_lengths=False,
                               # desc_only=False,
                               calc_uncertainty=True,
                               mc_samples=2,
                               desc_file_suffix_name='', nb_jobs=6, results_file=results_file)

    sys.exit()
