#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/10/16"

import sys, os.path

base_dir = os.path.dirname(os.path.abspath(__file__))
atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data'))
apt_dir = os.path.normpath(os.path.join(base_dir, "../../../apt/"))

if not common_dir in sys.path:
    sys.path.insert(0, common_dir)
    sys.path.insert(0, nomad_ml_dir)
    sys.path.insert(0, atomic_data_dir)
    sys.path.insert(0, apt_dir)

import numpy as np
import os
import math
from ai4materials.utils.utils_config import read_configs
from ai4materials.utils.utils_config import setup_logger
from ai4materials.wrappers import get_json_list
from ai4materials.wrappers import load_descriptor
from ai4materials.utils.utils_crystals import create_supercell, create_vacancies, random_displace_atoms, \
    get_structures_by_boxes
from ai4materials.utils.utils_data_retrieval import generate_facets_input
from ai4materials.utils.utils_plotting import plot_save_cnn_results
from ai4materials.models.cnn_nature_comm_ziletti2018 import model_deep_cnn_struct_recognition
from ai4materials.wrappers import calc_descriptor
from ai4materials.utils.utils_data_retrieval import write_ase_db
from ai4materials.utils.utils_plotting import aggregate_struct_trans_data, make_crossover_plot
from ai4materials.utils.utils_parsing import read_data
from functools import partial
from ai4materials.models.cnn_nature_comm_ziletti2018 import model_cnn_rot_inv, predict_cnn_keras, train_cnn_keras, model_fully_conv
from ai4materials.dataprocessing.preprocessing import make_data_sets
from ai4materials.descriptors.diffraction3d import Diffraction3D
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
from ai4materials.dataprocessing.preprocessing import prepare_dataset
import optparse

from datetime import datetime

startTime = datetime.now()
now = datetime.now()

# parser = optparse.OptionParser()
# parser.add_option('-p', '--np', action="store", dest="n_proc", help="number of processors", default=N_PROC)
# options, args = parser.parse_args()
# N_PROC = int(options.n_proc)
# N_PROC = 4

# read config file
config_file = '/home/ziletti/nomad/nomad-lab-base/analysis-tools/structural-similarity/config_default.yml'
configs = read_configs(config_file)

logger = setup_logger(configs, level='INFO', display_configs=False)

data_folder = [# '/home/ziletti/Documents/calc_xray/rot_inv_3d/prototypes_aflow_rot_inv/fe_only']
    # '/home/ziletti/Documents/calc_xray/rot_inv_3d/prototypes_aflow_rot_inv/A_hP2_194_c']  # ,
'/home/ziletti/Documents/calc_nomadml/rot_inv_3d/prototypes_aflow_rot_inv/A_cF4_225_a',
'/home/ziletti/Documents/calc_nomadml/rot_inv_3d/prototypes_aflow_rot_inv/A_cF8_227_a',
'/home/ziletti/Documents/calc_nomamld/rot_inv_3d/prototypes_aflow_rot_inv/A_cI2_229_a']


# add / at the end
main_folder = '/home/ziletti/Documents/calc_xray/rot_inv_3d/'
polycrystal_dir = '/home/ziletti/Documents/calc_xray/rot_inv_3d/polycrystals'

# directories
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
polycrystal_slices_dir = os.path.abspath(os.path.normpath(os.path.join(polycrystal_dir, 'slices')))

configs['io']['tmp_folder'] = tmp_folder
configs['io']['desc_folder'] = desc_folder
configs['io']['main_folder'] = main_folder
configs['io']['dataset_folder'] = dataset_folder

if not os.path.exists(polycrystal_slices_dir):
    os.makedirs(polycrystal_slices_dir)

# files
conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))
lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
training_log_file = os.path.abspath(
    os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))


    # os.path.abspath(os.path.normpath(os.path.join(polycrystal_dir, 'four_grains_poly.xyz')))
# os.path.normpath(os.path.join(polycrystal_dir, 'inclusions/AlSi_final.xyz')))
# polycrystal_file = os.path.normpath(os.path.join(main_folder, 'structural_defects/screw_dislocation/screw_dislocations_4.xyz'))
# polycrystal_file = os.path.normpath(os.path.join(main_folder, 'structural_defects/screw_dislocation/screw_dislocations_4_small.xyz'))
polycrystal_file = os.path.normpath(os.path.join(main_folder, 'structural_defects/crystal_twinning/fcc_crystal_twinning.xyz'))
#     os.path.normpath(os.path.join(polycrystal_dir, 'structural_defects/edge_dislocation/edge_dislocation.xyz')))
#     os.path.normpath(os.path.join(polycrystal_dir, 'four_grains_poly_large.xyz')))

# =============================================================================
#  Define descriptor
# =============================================================================
desc_names = [item.split("/")[-1] for item in data_folder]

user_param_source = {'wavelength': 4.0E-12,  # best overall
                     'pulse_energy': 1E-3, 'focus_diameter': 1E-3}

user_param_detector = {'distance': 0.1,  # on this scale it is not very important
                       'pixel_size': 3E-4, 'nx': 64, 'ny': 64}  # best for 3d --> pixel_size': 3E-4 with 64px

input_dims = (52, 32)

kwargs = dict(user_param_source=user_param_source, user_param_detector=user_param_detector, atoms_scaling='quantile_nn',
              extrinsic_scale_factor=1.00, atoms_scaling_cutoffs=[5.0, 7.0, 9.0, 11.0, 13.0, 15.0], use_mask=True,
              mask_r_min=12, mask_r_max=92, phi_bins=100, theta_bins=50, phi_bins_fine=512, theta_bins_fine=256,
              sph_l_cutoff=32, nx_fft=128, ny_fft=128, nz_fft=128)

descriptor = Diffraction3D(configs=configs, **kwargs)

# define operations on structures
operations_on_structure_list = [
    (create_supercell,
                                 dict(create_replicas_by='user-defined', target_replicas=[1, 1, 1],
                                     random_rotation=False)),
    (create_vacancies, dict(target_vacancy_ratio=0.25,
                            create_replicas_by='user-defined',
                            cell_type=None,
                            target_replicas=[1, 1, 1],
                            random_rotation=False,
                            optimal_supercell=False)),
    (random_displace_atoms, dict(displacement_scaled=0.01,
                            create_replicas_by='user-defined',
                            cell_type=None,
                            target_replicas=[1, 1, 1],
                            noise_distribution='uniform_scaled', target_nb_atoms=128, random_rotation=False,
                            optimal_supercell=False)),
    (create_supercell,
     dict(min_nb_atoms=32, target_nb_atoms=128, random_rotation=True, cell_type='standard', optimal_supercell=False)),

    (create_vacancies,
     dict(target_vacancy_ratio=0.25, create_replicas_by='nb_atoms', cell_type='standard', target_nb_atoms=128,
          random_rotation=True, optimal_supercell=True)),

    (random_displace_atoms, dict(displacement_scaled=0.05, create_replicas_by='nb_atoms', cell_type='standard',
                                 noise_distribution='uniform_scaled', target_nb_atoms=128, random_rotation=True,
                                 optimal_supercell=True)),

    (random_displace_atoms,
     dict(displacement=0.2, create_replicas_by='nb_atoms', cell_type='standard', noise_distribution='gaussian',
          target_nb_atoms=128, random_rotation=True, optimal_supercell=False)),

    (create_vacancies, dict(target_vacancy_ratio=0.25, create_replicas_by='user-defined', target_replicas=[1, 1, 1],
                            random_rotation=False)), (random_displace_atoms,
                                                      dict(displacement=0.1, create_replicas_by='user-defined',
                                                           noise_distribution='gaussian', target_replicas=[1, 1, 1],
                                                           random_rotation=False)), ]
# =============================================================================
# Descriptor calculation
# =============================================================================

# spgroups = ['194', '225', '227', '229']
spgroups = ['225', '227', '229']
# spgroups = ['227', '229']
# spgroups = ['194', '225']
# spgroups = ['194']
# spgroups = ['229']
# #
json_list = []
for idx in range(len(data_folder)):
    json_list = []
    logger.info("Calculating spgroup {}".format(spgroups[idx]))
    json_list.extend(
        get_json_list(method='folder', drop_duplicates=False, data_folder=data_folder[idx], tmp_folder=tmp_folder)[:2])

    ase_atoms_list = read_data(json_list, calc_spgroup=True, symprec=[1e-01, 1e-03])
#
# #   file_format='NOMAD' should be in write_json_for_nomad_sim
# # json_list = write_json_for_nomad_sim(output_folder=tmp_json_folder, ase_atom_list=ase_atoms_list, label_name='nmd_checksum')
#
    desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                     tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
                                     # desc_file='spgroup' + str(spgroups[idx]) + '_nb_atoms_128_disp01.tar.gz',
                                     # desc_file='spgroup' + str(spgroups[idx]) + '_nb_atoms_128_vac25.tar.gz',
                                     # desc_file='spgroup' + str(spgroups[idx]) + '_nb_atoms_128_pristine.tar.gz',
                                     desc_file='try.tar.gz',
                                     # desc_file='desc_4_classes.tar.gz',
                                     operations_on_structure=operations_on_structure_list[3],
                                     # operations_on_structure=None,
                                     nb_jobs=6, ** kwargs)
#
sys.exit()
# ase_atoms_list = read_data(json_list, calc_spgroup=True, symprec=[1e-01, 1e-03])
#
#  # file_format='NOMAD' should be in write_json_for_nomad_sim
# # json_list = write_json_for_nomad_sim(output_folder=tmp_json_folder, ase_atom_list=ase_atoms_list, label_name='nmd_checksum')

ase_atoms_list = get_structures_by_boxes(polycrystal_file, stride_size=[1., 1., 15.0], box_size=12.0,
                                         show_plot_lengths=True)


# desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
#                                  tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
#                                  # desc_file='spgroup'+str(spgroups[idx])+'_nb_atoms_128_opt_disp01.tar.gz',
#                                  # desc_file='spgroup' + str(spgroups[idx]) + '_nb_atoms_128_disp01.tar.gz',
#                                  # desc_file='spgroup' + str(spgroups[idx]) + '_nb_atoms_128_disp01_nfft192_lmax6.tar.gz',
#                                  desc_file='fcc_crystal_twinning_stride_6_vac25.tar.gz',
#                                  # desc_file='crystal_twinning_stride4_pristine.tar.gz',
#                                  # desc_file='four_grains_stride4_vac25_box13.tar.gz',
#                                  # desc_file='desc_4_classes.tar.gz',
#                                  # desc_file='try1' + '.tar.gz',
#                                  operations_on_structure=operations_on_structure_list[1],
#                                  # operations_on_structure=None,
#                                  nb_jobs=6, ** kwargs)

sys.exit()

# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/screw_dislocations_small_stride_2.tar.gz'
# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/screw_dislocations_small_stride_4.tar.gz'
desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/fcc_crystal_twinning_stride_6_vac25.tar.gz'

# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/screw_dislocations_stride_6.tar.gz'
# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/four_grains_pristine.tar.gz'
# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/crystal_twinning_stride_4.tar.gz'
# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/desc_4_classes.tar.gz',
# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/four_grains_stride4_vac25_box13.tar.gz'
# desc_files = '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/try1_spgroup194.tar.gz'


# desc_files = [
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/try1.tar.gz']
#
# target_list, structure_list = load_descriptor(desc_files=desc_files, configs=configs)

# df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_3d_sh_spectrum',
#                                          target_list=target_list, configs=configs, normalize=True)

# ase_db_file_pristine = write_ase_db(ase_atoms_list=ase_atoms_list,
#                                     db_name='spgroup225_nb_atoms_128_not_opt', main_folder=main_folder,
#                                     folder_name='db_ase')
# sys.exit(1)
#
# desc_file_path_1 = [
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/desc_4_classes.tar.gz']
# desc_file_path_2 = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/exp_fcc_aluminium.tar.gz']
#
# desc_files = desc_file_path_1 + desc_file_path_2


# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup194_nb_atoms_128_pristine.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup225_nb_atoms_128_pristine.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup227_nb_atoms_128_pristine.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup229_nb_atoms_128_pristine.tar.gz']
#
# desc_files = [
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup194_nb_atoms_128_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup225_nb_atoms_128_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup227_nb_atoms_128_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup229_nb_atoms_128_vac25.tar.gz']

# desc_files = [
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup194_nb_atoms_128_disp01.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup225_nb_atoms_128_disp01.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup227_nb_atoms_128_disp01.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup229_nb_atoms_128_disp01.tar.gz']

# desc_files = [
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup194_nb_atoms_256_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup225_nb_atoms_256_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup227_nb_atoms_256_vac25.tar.gz',
#     '/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/spgroup229_nb_atoms_256_vac25.tar.gz']


target_list, ase_atoms_list = load_descriptor(desc_files=desc_files, configs=configs)

# df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='sh_spectra',
#                                          target_list=target_list, configs=configs, normalize=True)

# sys.exit()
# =============================================================================
# Dataset preparation
# =============================================================================

new_labels = {"bct_139": ["139"], "bct_141": ["141"], "hcp": ["194"], "sc": ["221"], "fcc": ["1", "225"],
              "diam": ["227"], "bcc": ["229"]}

# new_labels=None
# path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(structure_list=ase_atoms_list,
#                                                                           target_list=target_list,
#                                                                           desc_metadata='diffraction_3d_sh_spectrum',
#                                                                           target_name='spacegroup_nb_symprec_0.001',
#                                                                           target_categorical=True,
#                                                                           dataset_folder=dataset_folder,
#                                                                           configs=configs,
#                                                                           dataset_name='rot_inv_pristine',
#                                                                           input_dims=input_dims,
#                                                                           main_folder=main_folder,
#                                                                           desc_folder=desc_folder,
#                                                                           tmp_folder=tmp_folder, disc_type=None,
#                                                                           n_bins=None,
#                                                                           notes="Prototypes bcc, fcc, diam, hcp.",
#                                                                           new_labels=new_labels)
# # #
# sys.exit()
path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(structure_list=ase_atoms_list,
                                                                       target_list=target_list,
                                                                       desc_metadata='diffraction_3d_sh_spectrum',
                                                                       # target_name='spacegroup_nb_symprec_0.001',
                                                                       target_name='target',
                                                                       target_categorical=True,
                                                                       dataset_folder=dataset_folder, configs=configs,
                                                                       # dataset_name='rot_inv_vac25',
                                                                       # dataset_name='screw_dislocations_small_stride_4',
                                                                       dataset_name='fcc_crystal_twinning_stride_6_vac25',
                                                                       input_dims=input_dims, main_folder=main_folder,
                                                                       desc_folder=desc_folder, tmp_folder=tmp_folder,
                                                                       disc_type=None, n_bins=None,
                                                                       notes="",
                                                                       new_labels=new_labels)
# sys.exit()
train_set_name = 'rot_inv_pristine'
# train_set_name = 'rot_inv_pristine'
# train_set_name = 'rot_inv_proto_bcc_fcc_diam_hcp_fft_with_scale'
# train_set_name = 'bcc_fcc_diam_hcp_disp01'
path_to_x_train = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')))
path_to_y_train = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')))
path_to_summary_train = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')))

# test_set_name = 'four_grains_stride4_vac25_box13'
# test_set_name = 'rot_inv_vac25'
# test_set_name = 'rot_inv_proto_bcc_fcc_diam_hcp_fft_with_scale'
# test_set_name = 'screw_dislocations_stride_6'
# test_set_name = 'screw_dislocations_small_stride_4'
test_set_name = 'fcc_crystal_twinning_stride_6_vac25'
# test_set_name = 'four_grains_stride4_vac25'
path_to_x_test = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_x.pkl')))
path_to_y_test = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_y.pkl')))
path_to_summary_test = os.path.abspath(
    os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_summary.json')))

x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                              path_to_summary=path_to_summary_train)

x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                           path_to_summary=path_to_summary_test)

params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
    "classes": dataset_info_train["data"][0]["classes"],
    # "checkpoint_filename": 'try_' + str(now.isoformat()),
    # "checkpoint_filename": 'try_2018-05-18T16:08:32.810714',
    "checkpoint_filename": 'small_nn_3d',  #best one so far
    # "checkpoint_filename": 'rot_inv_kernel_15',
    "batch_size": 32, "img_channels": 1}

# text_labels=np.asarray(dataset_info_train["data"][0]["text_labels"])
# numerical_labels=np.asarray(dataset_info_train["data"][0]["numerical_labels"])

text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

data_set_train = make_data_sets(x_train_val=x_train, y_train_val=y_train, split_train_val=True, test_size=0.1,
                                x_test=x_test, y_test=y_test, flatten_images=False)

# =============================================================================
# Neural network training and prediction
# =============================================================================

# partial_model_architecture = partial(model_deep_embedding_rot_inv,
#        conv2d_filters=[16, 12, 8, 8, 4, 4],
#        kernel_sizes=[3, 3, 3, 3, 3, 3],
#        max_pool_strides=[2, 2],
#        hidden_layer_size=64)

# this with 20 epochs is great
# partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[4, 4, 2, 2, 2, 2],
#                                      kernel_sizes=[9, 9, 9, 9, 9, 9], hidden_layer_size=32)

# partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[4, 4, 2, 2, 2, 2],
#                                      kernel_sizes=[15, 15, 15, 15, 15, 15], hidden_layer_size=32)

# partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[32, 32, 64, 64, 512, 2],
#                                      kernel_sizes=[3, 3, 3])

partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[8, 8, 8, 8, 4, 4],
                                     kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64)

# partial_model_architecture = partial(model_fully_conv, conv2d_filters=[32, 32, 16, 16, 8, 512],
#                                      kernel_sizes=[3, 3, 3, 3, 3])

# partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[4, 4, 2, 2, 2, 2],
#                                      kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=32)
# beautiful maps
#        conv2d_filters=[32, 16, 12, 8, 4, 4],
#        kernel_sizes=[3, 3, 3, 3, 3, 3], 

#        conv2d_filters=[32, 16, 12, 8, 4, 4],
#        kernel_sizes=[5, 5, 5, 5, 5, 5], 
# partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[16, 12, 8, 8, 4, 4],
#                                      kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=32)

# train_cnn_keras(data_set=data_set_train, configs=configs, nb_classes=params_cnn["nb_classes"],
#                 partial_model_architecture=partial_model_architecture, batch_size=params_cnn["batch_size"],
#                 img_channels=1, checkpoint_dir=checkpoint_dir, checkpoint_filename=params_cnn["checkpoint_filename"],
#                 nb_epoch=100, training_log_file=training_log_file, early_stopping=False, data_augmentation=False,
#                 normalize=True)

data_set_predict = make_data_sets(x_train_val=x_test, y_train_val=y_test, split_train_val=False, test_size=0.1,
                                  x_test=x_test, y_test=y_test)

target_pred_class, target_pred_probs, prob_predictions, conf_matrix = predict_cnn_keras(data_set_predict,
                                                                                        params_cnn["nb_classes"],
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
                                                                                        results_file=results_file,
                                                                                        normalize=True)

prob_0 = prob_predictions[0:, 0].tolist()
prob_1 = prob_predictions[0:, 1].tolist()
prob_2 = prob_predictions[0:, 2].tolist()
prob_3 = prob_predictions[0:, 3].tolist()

# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/four_grains_stride4_vac25_box13.tar.gz']
# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/four_grains_stride4_vac40.tar.gz']
# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/screw_dislocations_stride_6.tar.gz']
# desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/screw_dislocations_small_stride_2.tar.gz']
desc_files = ['/home/ziletti/Documents/calc_xray/rot_inv_3d/desc_folder/fcc_crystal_twinning_stride_6_vac25.tar.gz']

target_list, structure_list = load_descriptor(desc_files=desc_files, configs=configs)
#
generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_3d_sh_spectrum', target_list=target_list,
                      make_sprite_atlas=False, configs=configs, normalize=True, prob_0=prob_0, prob_1=prob_1,
                      prob_2=prob_2, prob_3=prob_3)

# plot_save_cnn_results(training_log_file, accuracy=True, cross_entropy_loss=True,
#   show_plot=True)

sys.exit(1)

### plot neural network training log
# plot_save_cnn_results(training_log, accuracy=True, cross_entropy_loss=True,
#   show_plot=True)


# df_results = aggregate_struct_trans_data(results_file_bcc_to_amorphous,
#    nb_rows_to_cut=8,
#    nb_samples=1,
#    nb_order_param_steps=16, max_order_param=0.95, 
#    # extract all classes
#    prob_idxs=range(params_cnn["nb_classes"]))
#
# print "classes", params_cnn["classes"]
## classes [u'bcc', u'bct_139', u'bct_141', u'diam', u'fcc', u'hex/rh', u'sc']
#
# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"],
#    nb_order_param_steps=20,     
#    filename_suffix=".png", 
#    title="From body-centered-cubic (bcc) to amorphous", 
#    x_label="Vacancies (atoms removed) [%]", show_plot=False)
#
# sys.exit(1)


# =============================================================================
# MISCLASSIFICATION PLOTTING
# =============================================================================

# get classes from spacegroup value
# The classes for the Viewer needs to start from 0 onwards (integers)
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(class_labels)
# target_class = le.transform(class_labels)
# classes = list(le.classes_)
# target_class = [int(item) for item in target_class]
##
#
#    
## make lists of single elementes
# new_target_list = []
# for idx, item in enumerate(target_class):
#    new_target_list.append([target_class[idx]]) 
#
# target_list = new_target_list
#
# checkpoint_filename='try_all_classes_kernel_2017-10-16T10:55:29.694457'
#
# target_files = []
#
# desc_file_list = desc_file_list_test
#
# for desc_file in desc_file_list:
#    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))
#    target = extract_target_files(filename=desc_file_path, desc_folder=desc_folder, tmp_folder=tmp_folder)   
#    target_files.append(target)     
#
## flatten the list
# target_files_list = [item for sublist in target_files for item in sublist]
#
# json_list = []
# for idx, item in enumerate(target_files_list):
##        print item["data"][0]["main_json_file_name"], images_filelist[idx].split("/")[-1]
#    # the 0 refers to the first frame
#    json_list.append(item["data"][0]["main_json_file_name"])
#    
# target_pred_list = target_pred_probs
# target_list = num_labels
#
# plot_misclassified_only = True
#
# desc_file_list = ['descriptor_all_classes_8_samples.tar.gz']
# checkpoint_filename='try_2017-11-15T10:37:39.930223'
#
# 
# for desc_file in desc_file_list:
##    
###    desc_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'descriptor_rnh_all.tar.gz')))
##
#    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))
##    
#    desc_images = extract_images(filename=desc_file_path, 
#        filetype='descriptor_files',
#        input_dims=input_dims, desc_folder=desc_folder, tmp_folder=tmp_folder)
####
###    
# images = desc_images
# model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename +'.h5')))
# model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename + '.json')))
##
#
# print "images.shape", images.shape
#
# plot_att_response_maps(images, model_arch_file, model_weights_file, figure_dir,
#    nb_conv_layers=6,
#    nb_top_feat_maps=4, 
#    layer_nb='all',
##    layer_nb=[0, 7],
#    plot_all_filters=False,
#    plot_filter_sum=True,
#    plot_summary=True)
#
#

print "Execution time: ", datetime.now() - startTime

# if plot_misclassified_only:
#    # add mask to plot only misclassified samples    
#    assert len(target_class) == len(target_pred_class)
#    
#    misclassied_list = []
#    for idx, item in enumerate(target_pred_list):
#        if target_class[idx] == target_pred_class[idx]:        
#            misclassied_list.append(False)
#        else:
#            misclassied_list.append(True)
#        
#    target_list = [target_list[i] for i in xrange(len(target_list)) if misclassied_list[i]]
#    target_pred_list = [target_pred_list[i] for i in xrange(len(target_pred_list)) if misclassied_list[i]]
#    json_list = [json_list[i] for i in xrange(len(json_list)) if misclassied_list[i]]
#
#    print "misclassified list", json_list
#
#
# df_filepath = generate_facets_input(desc_folder=desc_folder, main_folder=main_folder,
#    input_dims=input_dims, desc_file_list=desc_file_list, tmp_folder=tmp_folder,
#    misclassified_list=misclassied_list)
##
#
# sys.exit(1)

# desc_file_list = ['descriptor_all_classes_8_samples.tar.gz']


# desc_file_list = ['descriptor_try1.tar.gz']


#
# for desc_file in desc_file_list:
##    
###    desc_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'descriptor_rnh_all.tar.gz')))
##
#    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))
##    
#    desc_images = extract_images(filename=desc_file_path, 
#        filetype='descriptor_files',
#        input_dims=input_dims, desc_folder=desc_folder, tmp_folder=tmp_folder)
###
##    
# images = desc_images
##model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] +'.h5')))
##model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] + '.json')))
##
#
# print "images.shape", images.shape

# plot_att_response_maps(images, model_arch_file, model_weights_file, figure_dir,
#    nb_conv_layers=6,
#    nb_top_feat_maps=8, 
#    layer_nb='all',
#    layer_nb=[0],
#    plot_all_filters=True,
#    plot_filter_sum=True,
#    plot_summary=True)
#
#

print "Execution time: ", datetime.now() - startTime

# sys.exit(1)

# read panda dataframe and plot results
# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv')))
# plot_bcc_to_scc(results_file, show_plot=True)

# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_diamond_fcc.csv')))
# plot_diamond_to_fcc(results_file, show_plot=True)

# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_bcc_disorder_all.csv')))
# plot_bcc_disorder(results_file, show_plot=True)

print "Execution time: ", datetime.now() - startTime

# sys.exit(1)

# desc_file_list = ['descriptor_4_classes_194_225_227_229.tar.gz',
#                'descriptor_slices_poly.tar.gz']

# desc_file_list = ['descriptor_slices_poly.tar.gz']

# desc_file_list = ['descriptor_slices_try1.tar.gz']

desc_file_list = ['descriptor_slices_fcc_diam_fft_pristine_slide1.tar.gz']

# desc_file_list=['descriptor_proto_all.tar.gz']
# desc_file_list = ['descriptor_try_proto_try_defects.tar.gz']

target_pred_probs = None
num_labels = None
target_pred_list = target_pred_probs
target_list = num_labels

#
#

model_weights_file = os.path.abspath(
    os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] + '.h5')))
model_arch_file = os.path.abspath(
    os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] + '.json')))
#

clustering_labels = calc_clustering(desc_type='xray_3d',  # lookup_file=lookup_file, desc_file=desc_file_list_test[0],
                                    target_categorical=True, input_dims=input_dims, lookup_file=lookup_file,
                                    desc_file=desc_file_list,  # desc_folder=tmp_folder,
                                    desc_folder=desc_folder, standardize='True',  # standardize='variance',
                                    #    standardize='False',
                                    target_name='target', use_xray_img=True, model_arch_file=model_arch_file,
                                    model_weights_file=model_weights_file, nb_nn_layer=-4, batch_size=1000,
                                    use_dist=True)

# desc_file_list = ['descriptor_slices_fcc_diam_fft_vac40_slide2.tar.gz']

# for desc_file in desc_file_list:
#     desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))

# df_filepath = generate_facets_input(desc_folder=desc_folder, main_folder=main_folder,
#     input_dims=input_dims, desc_file_list=desc_file_list, tmp_folder=tmp_folder,
#     normalize=True, clustering_labels=clustering_labels)

# sys.exit(1)

# 1000 is the best
embed_params = {'learning_rate': 100, 'method': 'exact', 'perplexity': 10, 'n_iter': 5000,  # 'metric': 'precomputed'
                }

text_labels = clustering_labels
## embedding is not compatible with desc_file being a list
# calc_embedding(embed_method='hessian', desc_type='xray_3d',
# calc_embedding(embed_method='isomap', desc_type='xray_3d',
# calc_embedding(embed_method='pca', desc_type='xray_3d',
calc_embedding(embed_method='tsne_pca', desc_type='xray_3d',  # calc_embedding(embed_method='tsne', desc_type='xray_3d',
               # calc_embedding(embed_method='spectral_embedding', desc_type='xray_3d',
               # calc_embedding(embed_method='mds', desc_type='xray_3d',
               #    lookup_file=lookup_file, desc_file=desc_file_list_test[0],
               target_categorical=True, input_dims=input_dims, lookup_file=lookup_file, desc_file=desc_file_list,
               # desc_folder=tmp_folder,
               desc_folder=desc_folder, standardize='True',  # standardize='variance',
               #  standardize='False',
               target_name='target', embed_params=embed_params, use_xray_img=False, model_arch_file=model_arch_file,
               model_weights_file=model_weights_file, nb_nn_layer=-4, batch_size=1000, use_dist=False)
#    
#

json_list, frame_list, x_list, y_list, foo = get_json_list(method='file', data_folder=data_folder,
                                                           # path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
                                                           path_to_file=lookup_file, drop_duplicates=False,
                                                           displace_duplicates=False, get_unique_list=True)

'''
json_list = []
frame_list = []
x_list = []
y_list = []

# the target list is the energy, but we want the classification
for folder in data_folder:
    json_list_, frame_list_, x_list_, y_list_, foo = get_json_list(method='file', data_folder=folder,
        #path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
        path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
        
    json_list.append(json_list_)
    frame_list.append(frame_list_)
    x_list.append(x_list_)
    y_list.append(y_list_)


json_list = [item for sublist in json_list for item in sublist]
frame_list = [item for sublist in frame_list for item in sublist]
x_list = [item for sublist in x_list for item in sublist]
y_list = [item for sublist in y_list for item in sublist]
'''

op_list = []
for i in range(len(json_list)):
    op_list_ = []
    for j in range(len(operations_on_structure_list)):
        op_list_.append(j)

    op_list.append(op_list_)

op_list = [item for sublist in op_list for item in sublist]

'''
# if target_list = num_labels
# we do not need to duplicate the target list according to the number of 
# operations
# duplicate the target list according to the number of operations
new_target_list = []
for i in range(len(target_list)):
    new_target_list_ = []
    for j in range(len(operations_on_structure)):
        new_target_list_.append(target_list[i])

    new_target_list.append(new_target_list_)
    
target_list = new_target_list
target_list = [item for sublist in target_list for item in sublist]

'''

# get classes from spacegroup value
# the classes for the Viewer needs to start from 0 onwards (integers)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(text_labels)
target_class = le.transform(text_labels)
classes = list(le.classes_)
target_class = [int(item) for item in target_class]
#


# make lists of single elementes
new_target_list = []
for idx, item in enumerate(target_class):
    new_target_list.append([target_class[idx]])

target_list = new_target_list

# make list for each element
frame_list = [item for sublist in frame_list for item in sublist]
frame_list = [[item] for item in frame_list]

try:
    x_list = [item for sublist in x_list for item in sublist]
    y_list = [item for sublist in y_list for item in sublist]
except:
    pass

# print 'original target list', target_list, len(target_list)

# read from desc_file
# try:
#    with open(desc_info_file) as data_file:
#        data = json.load(data_file)
#
#        for c in data['descriptor_info']:
#            xray_img_list = c["xray_img_list"]
# except:
#    xray_img_list = None


xray_img_list = []

for desc_file_list_test_ in desc_file_list:
    xray_member_list = []
    with tarfile.open(os.path.join(desc_folder, desc_file_list_test_), 'r') as archive:
        for member in archive.getmembers()[:100000]:
            if not member.isfile():
                continue
            if member.name.endswith('_xray.png'):
                xray_img_list.append(member.name)
                xray_member_list.append(member)

        # it extract all the files but that's okay for now
        archive.extractall(path=desc_folder, members=xray_member_list)

xray_img_list = [os.path.join(desc_folder, item) for item in xray_img_list]

# for idx, item in enumerate(target_class):
#    print idx, target_list[idx], target_pred_class[idx]


plot_misclassified_only = False
target_pred_class = None

if plot_misclassified_only:
    # add mask to plot only misclassified samples    
    assert len(target_class) == len(target_pred_class)

    misclass_mask = []
    for idx, item in enumerate(target_pred_list):
        if target_class[idx] == target_pred_class[idx]:
            misclass_mask.append(False)
        else:
            misclass_mask.append(True)

    target_list = [target_list[i] for i in xrange(len(target_list)) if misclass_mask[i]]
    target_pred_list = [target_pred_list[i] for i in xrange(len(target_pred_list)) if misclass_mask[i]]
    json_list = [json_list[i] for i in xrange(len(json_list)) if misclass_mask[i]]
    frame_list = [frame_list[i] for i in xrange(len(frame_list)) if misclass_mask[i]]
    op_list = [op_list[i] for i in xrange(len(op_list)) if misclass_mask[i]]
    x_list = [x_list[i] for i in xrange(len(x_list)) if misclass_mask[i]]
    y_list = [y_list[i] for i in xrange(len(y_list)) if misclass_mask[i]]
    xray_img_list = [xray_img_list[i] for i in xrange(len(xray_img_list)) if misclass_mask[i]]

    print json_list

# only for visualization
# by default the operations defined before the descriptor calculation are used    
# operations_on_structure = [
##    (create_vacancies, {'vacancy_ratio': 0.1, 'replicas': [3, 3, 3]})]
##    (random_displace_atoms, {'displacement': 0.20, 'replicas': [2, 2, 2]})]
#    (create_supercell, {'replicas': [1, 1, 1]})]

filename = plot(name='xray_plot', json_list=json_list, frames='list', frame_list=frame_list, op_list=op_list,
                descriptor=None, operations_on_structure=operations_on_structure_list, xray_img_list=xray_img_list,
                file_format='NOMAD', clustering_x_list=x_list, clustering_y_list=y_list, target_list=target_list,
                is_classification=True, target_class_names=classes, target_pred_list=target_pred_list, target_unit='',
                clustering_point_size=12, tmp_folder=tmp_folder, control_file=control_file, cell_type='standard',
                atoms_scaling='avg_distance_nn')

# =============================================================================
# OLD STUFF
# 
# =============================================================================

# accepted_labels=[
##[[63]],
##[[136]],
##[[148]],
##[[205]],
##[[217]]
##    [[134]],
#    [[139]],
#    [[141]],
##    [[152]],
#    [[166]],
#    [[194]],
#    [[221]],
#    [[225]],
#    [[227]],
#    [[229]]
#    ]

# start_desc=0
# for i in range(start_desc, len(desc_names)):
#
#    for accepted_label in accepted_labels:
#        json_list = []
#        
#        json_list.append(get_json_list(method='folder', drop_duplicates=False, 
#            data_folder=data_folder[i], tmp_folder=tmp_folder))
#            
#        json_list = [item for sublist in json_list for item in sublist]
#        #[:100] will give already the four classes with Rnh
##        json_list = json_list[:20]
#    
#        # filtering the json_list
#        json_list = filter_json_list(file_format='NOMAD',
#        json_list=json_list, tmp_folder=tmp_folder,
#        desc_folder=desc_folder,
#        #desc_folder=tmp_folder,
#    #    filter_by=['lattice_type', 'spacegroup_symbol'],
#        filter_by=['spacegroup_number'],
#    
#    #    filter_by=['lattice_type'],
#        cell_type=cell_type,
#        accepted_labels=accepted_label,
#    #    accepted_labels=[['cubic'], ['Fd-3m', 'Fm-3m', 'Im-3m', 'Pm-3m']],
#    #    ['I-43m', 'P2_13', 'Pa3']
#    #    accepted_labels=[['cubic'], ['I-43m']],#, 'P2_13', 'Pa3']],
#    #    accepted_labels=[['cubic'], ['P2_13']],
#    #    accepted_labels=[['cubic'], ['Pa3']],
#    
#    #, 'P2_13', 'Pa3']],
#    #    accepted_labels=[['cubic'], ['Im-3m']],
#    #    accepted_labels=[['cubic'], ['Fd-3m']],
#    
#    #    accepted_labels=[['cubic'], ['Fd-3m', 'Im-3m']],
#    
#    #    accepted_labels=[['cubic'], ['Pm-3m']],
##        write_to_file=False,
#        write_to_file=True,
#        filtered_file=desc_names[i] + '_spgroup' + str(accepted_label[0][0]) + '.json.filter',
#        operations_on_structure=operations_on_structure,
#        **kwargs) 
#    #       
#    #    operations_on_structure=operations_on_structure,
#    #    **kwargs) 
##        print 'calculating descriptor', i
##        calc_descriptor(desc_type='xray', file_format='NOMAD',
##            json_list=json_list, tmp_folder=tmp_folder,
##            desc_folder=desc_folder,
##            #desc_folder=tmp_folder,
###            desc_file='descriptor_all_134.tar.gz',
##            desc_file=desc_names[i]+'_spgroup'+str(accepted_label[0][0])+'_supercell_by_nb_atoms_min16_max128_mask_r5_rot_inv_no_norm'+'.tar.gz',
###            desc_file=desc_names[i]+'_spgroup'+str(accepted_label[0][0])+'_supercell_by_nb_atoms_min16_max128_vacancy_20'+'.tar.gz',
##
##    #        desc_file=desc_names[i]+'_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40'+'.tar.gz',
##    #        desc_file='descriptor'+'.tar.gz',
##            desc_info_file=desc_info_file,
##            grayscale=True,
##            # stupid but works
##            target_list=np.zeros(len(json_list)), 
##            cell_type=cell_type,
##            operations_on_structure=operations_on_structure,
##            **kwargs)
