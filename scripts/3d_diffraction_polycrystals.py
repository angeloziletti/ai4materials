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
    from ai4materials.descriptors.diffraction3d import DISH
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.models.strided_pattern_matching import get_classification_map
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.models.strided_pattern_matching import make_strided_pattern_matching_dataset
    from argparse import ArgumentParser
    from datetime import datetime

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
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models/enc_dec_drop12.5')))
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
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/four_grains/fcc_crystal_twinning.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/stacking_fault/Al_SF_large.xyz'))
    structure_files.append(os.path.join(main_folder, 'structures_for_paper/four_grains/four_grains_poly.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/four_grains/four_grains_poly_disp01_vac20.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/four_grains/four_grains_poly_disp04_vac50.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/grain_boundaries/0012262150_v6bxv2_tv0.4bxv0.3_d2.1z_traj.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/edge_dislocation/Al_edge.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/small_edge_dislocation/small_edge_dislocation.xyz'))
    # structure_files.append(os.path.join(main_folder, 'structures_for_paper/edge_dislocation/Al_edge_vac20.xyz'))

    descriptor = DISH(configs=configs)

    operations_on_structure_list = [
        (create_supercell, dict(create_replicas_by='user-defined', target_replicas=[1, 1, 1], random_rotation=False)), (
            create_vacancies, dict(target_vacancy_ratio=0.20, create_replicas_by='user-defined', cell_type=None,
                                   target_replicas=[1, 1, 1], random_rotation=False, optimal_supercell=False)), (
            random_displace_atoms,
            dict(displacement_scaled=0.01, create_replicas_by='user-defined', cell_type=None, target_replicas=[1, 1, 1],
                 noise_distribution='uniform_scaled', target_nb_atoms=128, random_rotation=False,
                 optimal_supercell=False))]

    # =============================================================================
    # Descriptor calculation
    # =============================================================================
    # stride_size = [0.5, 0.5, 20.0]
    stride_size = [6.0, 6.0, 20.0]
    box_sizes = [15.0, 15.0, 15.0, 15.0]
    padding_ratios = [(0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    # desc_file = None

    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac50.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac70.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac80.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/inclusions/bcc_fcc_inclusion_vac90.xyz_stride_1.5_1.5_1.5_box_size_13.5_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/grain_boundaries/0012262150_v6bxv2_tv0.4bxv0.3_d2.1z_traj.xyz_stride_1.0_1.0_1.0_box_size_10.0_.tar.gz')

    # desc_file = os.path.join(main_folder, 'desc_folder/fcc_crystal_twinning/fcc_crystal_twinning.xyz_stride_1.0_1.0_20.0_box_size_14.0_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/fcc_crystal_twinning/fcc_crystal_twinning.xyz_stride_0.5_0.5_20.0_box_size_10.0_.tar.gz')

    # desc_file = os.path.join(main_folder, 'desc_folder/stacking_fault/Al_SF_large.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/stacking_fault/Al_SF_large_vac50.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/stacking_fault/Al_SF_large.xyz_stride_0.5_0.5_20.0_box_size_10.0_.tar.gz')

    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_40.0_9.0_20.0_box_size_12.0_pristine.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_6.0_6.0_20.0_box_size_15.0_pristine.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_6.0_6.0_20.0_box_size_15.0_vac20.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_6.0_6.0_20.0_box_size_15.0_disp01.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_3.0_3.0_20.0_box_size_15.0_pristine.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly.xyz_stride_1.0_1.0_20.0_box_size_15.0_pristine.tar.gz')
    desc_file = os.path.join(main_folder,
                             'desc_folder/four_grains/four_grains_poly.xyz_stride_1.0_1.0_20.0_box_size_15.0_pristine.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp01_vac20.xyz_stride_6.0_6.0_20.0_box_size_15.1_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp4_vac50.xyz_stride_6.0_6.0_20.0_box_size_15.1.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp4_vac50.xyz_stride_1.0_1.0_20.0_box_size_15.1.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp2_vac50.xyz_stride_1.0_1.0_20.0_box_size_15.1.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp4_vac50.xyz_stride_1.0_1.0_20.0_box_size_15.1.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/four_grains/four_grains_poly_disp1_vac20.xyz_stride_1.0_1.0_20.0_box_size_15.1.tar.gz')

    # desc_file = os.path.join(main_folder,
    #                          'desc_folder/edge_dislocation/Al_edge_vac20.xyz_stride_1.0_1.0_20.0_box_size_18.1_.tar.gz')
    # desc_file = os.path.join(main_folder,
    #                          'desc_folder/edge_dislocation/Al_edge.xyz_stride_1.0_1.0_20.0_box_size_18.1.tar.gz')

    # desc_file = os.path.join(main_folder, 'desc_folder/small_edge_dislocation/small_edge_dislocation.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/small_edge_dislocation/small_edge_dislocation_vac10.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz')
    # desc_file = os.path.join(main_folder, 'desc_folder/small_edge_dislocation/small_edge_dislocation_disp1_vac10.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz')

    for idx, structure_file in enumerate(structure_files):
        # path_to_x_test, path_to_y_test, path_to_summary_test, path_to_strided_pattern_pos = make_strided_pattern_matching_dataset(
        #     polycrystal_file=structure_file, descriptor=descriptor, desc_metadata='diffraction_3d_sh_spectrum',
        #     configs=configs, operations_on_structure=None, stride_size=(10., 10., 20.), box_size=10.,
        #     init_sliding_volume=(14., 14., 14.), desc_file=None, desc_only=False, show_plot_lengths=True,
        #     desc_file_suffix_name='', nb_jobs=6, padding_ratio=None)

        path_to_x_test = os.path.join(dataset_folder, 'four_grains_poly_disp01_vac20.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz_x.pkl')
        path_to_y_test = os.path.join(dataset_folder, 'four_grains_poly_disp01_vac20.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar.gz_y.pkl')
        path_to_summary_test = os.path.join(dataset_folder,
                                            'four_grains_poly_disp01_vac20.xyz_stride_1.0_1.0_20.0_box_size_10.0_.tar'
                                            '.gz_summary.json')
        path_to_strided_pattern_pos = os.path.join(dataset_folder, 'four_grains_poly.xyz_stride_1.0_1.0_20.0_box_size_10.0__pristine.tar.gz_strided_pattern_pos.pkl')

        # path_to_x_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_SF_large.xyz_stride_6.0_6.0_20.0_box_size_15.1_.tar.gz_x.pkl'
        # path_to_y_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_SF_large.xyz_stride_6.0_6.0_20.0_box_size_15.1_.tar.gz_y.pkl'
        # path_to_summary_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_SF_large.xyz_stride_6.0_6.0_20.0_box_size_15.1_.tar.gz_summary.json'
        # path_to_strided_pattern_pos = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_SF_large.xyz_stride_6.0_6.0_20.0_box_size_15.1_.tar.gz_strided_pattern_pos.pkl'

        # path_to_x_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_edge.xyz_stride_4.0_4.0_4.0_box_size_12.0__pristine.tar.gz_x.pkl'
        # path_to_y_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_edge.xyz_stride_4.0_4.0_4.0_box_size_12.0__pristine.tar.gz_y.pkl'
        # path_to_summary_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_edge.xyz_stride_4.0_4.0_4.0_box_size_12.0__pristine.tar.gz_summary.json'
        # path_to_strided_pattern_pos = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/Al_edge.xyz_stride_4.0_4.0_4.0_box_size_12.0__pristine.tar.gz_strided_pattern_pos.pkl'

        # path_to_x_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/small_edge_dislocation.xyz_stride_10.0_10.0_20.0_box_size_10.0_.tar.gz_x.pkl'
        # path_to_y_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/small_edge_dislocation.xyz_stride_10.0_10.0_20.0_box_size_10.0_.tar.gz_y.pkl'
        # path_to_summary_test = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/small_edge_dislocation.xyz_stride_10.0_10.0_20.0_box_size_10.0_.tar.gz_summary.json'
        # path_to_strided_pattern_pos = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/datasets/small_edge_dislocation.xyz_stride_10.0_10.0_20.0_box_size_10.0_.tar.gz_strided_pattern_pos.pkl'

        get_classification_map(configs, path_to_x_test, path_to_y_test, path_to_summary_test,
                               path_to_strided_pattern_pos, checkpoint_dir, checkpoint_filename='model.h5',
                               mc_samples=2, interpolation='none', results_file=None, calc_uncertainty=True,
                               conf_matrix_file=conf_matrix_file, train_set_name='hcp-sc-fcc-diam-bcc_pristine',
                               cmap_uncertainty='hot', interpolation_uncertainty='none')
