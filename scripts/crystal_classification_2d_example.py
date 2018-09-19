#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/10/16"

import sys
import os.path

base_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.normpath(os.path.join(base_dir, "../../../python-common/common/python"))
nomad_sim_dir = os.path.normpath(os.path.join(base_dir, "../python-modules/"))
atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data'))
apt_dir = os.path.normpath(os.path.join(base_dir, "../../../apt/"))

if not common_dir in sys.path:
    sys.path.insert(0, common_dir)
    sys.path.insert(0, nomad_sim_dir)
    sys.path.insert(0, atomic_data_dir)
    sys.path.insert(0, apt_dir)

import numpy as np
import math
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger
from ai4materials.utils.utils_data_retrieval import read_ase_db
from ai4materials.models.cnn_nature_comm_ziletti2018 import train_neural_network
from ase.spacegroup import get_spacegroup
from ai4materials.models.cnn_nature_comm_ziletti2018 import predict
from ai4materials.models.cnn_architectures import cnn_nature_comm_ziletti2018
from functools import partial
from sklearn.model_selection import StratifiedShuffleSplit

from ai4materials.utils.utils_plotting import aggregate_struct_trans_data, make_crossover_plot
from ai4materials.utils.utils_parsing import read_data

from ai4materials.dataprocessing.preprocessing import make_data_sets
from ai4materials.dataprocessing.preprocessing import prepare_dataset
from ai4materials.descriptors.diffraction2d import Diffraction2D
from ai4materials.descriptors.diffraction3d import Diffraction3D
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
import optparse

from datetime import datetime

startTime = datetime.now()
now = datetime.now()

# N_PROC = 1
# parser = optparse.OptionParser()
# parser.add_option('-p', '--np', action="store", dest="n_proc", help="number of processors", default=N_PROC)
# options, args = parser.parse_args()


# read config file
config_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/config_default.yml'
# config_file = '/u/ziang/nomad/nomad-lab-base/analysis-tools/structural-similarity/config_eos.yml'
configs = set_configs(config_file)

logger = setup_logger(configs, level='DEBUG', display_configs=False)

data_folder = ['/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Rnh_4DFTJQgTSOib4e4d-5GByiTVB',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/R10ncY1AZG6X9y-Nj8F0_DiN8NeLD',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RsLoZhSAdK0BopfI2T4B5pLfMyjVN',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RMGpPc3B_HiR0D-oLE4ND66HmYdH-',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Re2mnhOAs6ZNqvTY1p-W2RavinjOM',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/R9usAWjw2xq9F8zW-66jyCyeDLlDa',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RkxmUCgPxt-9xDdIpr5xqPQK8PC9H',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RdzeezGR0W5wGEpGYEqOq7AygYS9J',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Rc_XxYadb0ZlfBVLqCNo-EtVocxv8',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RKXqE9xPCiLlufNK0n4pbtzdbID5H',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RYvdvBLf1QdM5QJ_8DVve7CknkdK5',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RUt3qcReY6SJO6fIJ5jangTSlMjaQ',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/REg0D-KojGnrw51EYl2Q2rCYOIfJM',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RSkoltrNkpZwp1xpi_Oj1jO4IndC5',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RA-tqhSlH5idfPW_3UxE80I7BBL6s',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RayT1o-XjyZaWdlVS_Fk8nssdO1w9',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r',
               '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Ra8nAuJOgxGwSytw1scU5BTeB3ozo']

data_folder = ['/home/ziletti/Documents/calc_xray/2d_nature_comm/rh_bcc_sc_fcc_rh']

# data_folder = [
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_tI2_139_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_tI4_141_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_hR1_166_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_hP2_194_c',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_cP1_221_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_cF4_225_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_cF8_227_a',
#     '/home/ziletti/Documents/calc_xray/2d_nature_comm/prototypes_aflow/A_cI2_229_a'
# ]

# add / at the end
main_folder = '/home/ziletti/Documents/calc_nomadml/2d_nature_comm/'
# main_folder = '/scratch/ziang/2d_nature_comm/'

# directories
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))

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
results_file_bcc_to_sc = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_crossover_bcc_sc.csv')))
results_file_diam_to_fcc = os.path.abspath(
    os.path.normpath(os.path.join(main_folder, 'results_crossover_diam_to_fcc.csv')))
results_file_bcc_to_amorphous = os.path.abspath(
    os.path.normpath(os.path.join(main_folder, 'results_crossover_bcc_to_amorphous.csv')))
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results_crossover.csv')))

# =============================================================================
#  Define descriptor
# =============================================================================
desc_names = [item.split("/")[-1] for item in data_folder]

user_param_source = {'wavelength': 5.0E-12,  # best overall
                     'pulse_energy': 1E-6, 'focus_diameter': 1E-6}

user_param_detector = {'distance': 0.1,  # on this scale it is not very important
                       'pixel_size': 4E-4, 'nx': 64, 'ny': 64}


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


# desc_angles = {"r": [-45., 45.], "g": [-45., 45.], "b": [-45., 45.]}
desc_angles = {"r": [-45., 45.], "g": [-135., 135.], "b": [-45., 45.]}

rot_matrices = {}
rot_matrices_x = []
for angle in desc_angles["r"]:
    rot_matrices_x.append(rot_mat_x(angle))
rot_matrices["r"] = rot_matrices_x

rot_matrices_y = []
for angle in desc_angles["g"]:
    rot_matrices_y.append(rot_mat_y(angle))
rot_matrices["g"] = rot_matrices_y

rot_matrices_z = []
for angle in desc_angles["b"]:
    rot_matrices_z.append(rot_mat_z(angle))
rot_matrices["b"] = rot_matrices_z

input_dims = (64, 64)

kwargs = dict(mask_r_min=5, user_param_source=user_param_source, user_param_detector=user_param_detector,
              atoms_scaling='avg_nn', use_mask=True, rot_matrices=rot_matrices,
              atoms_scaling_cutoffs=[4.0, 5.0, 7.0, 9.0, 11.0, 12.0])
# min_nn
# quantile_nn
# avg_nn
# kwargs = dict(user_param_source=user_param_source, user_param_detector=user_param_detector,
#               atoms_scaling='avg_distance_nn', atoms_scaling_cutoffs=[4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 20.0, 30.0],
#               use_mask=True, n_fft=128, mask_r_min=12, phi_bins=100, theta_bins=50, phi_bins_fine=256,
#               theta_bins_fine=256, sph_l_cutoff=32)

# descriptor = Diffraction3D(configs=configs, **kwargs)
# descriptor = Diffraction2D(configs=configs, **kwargs)
descriptor = Diffraction2D(configs=configs)


# descriptor = Diffraction1D(configs=configs, **kwargs)

# this works
# operations_on_structure_list = [(create_supercell_by_nb_atoms,
#                                  dict(min_nb_atoms=32, target_nb_atoms=256, random_rotation=False, random_rotation_before=True,
#                                 cell_type='standard', optimal_supercell=True))]#,

# define operations on structures
# operations_on_structure_list = [
# # (create_supercell,
# #                                  dict(create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
# #                                       random_rotation=False, random_rotation_before=True,
# #                                       cell_type='standard_no_symmetries', optimal_supercell=False))]  # ,
# (create_vacancies,
#  dict(target_vacancy_ratio=0.01, create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
#       random_rotation=False, random_rotation_before=True,
#       cell_type='standard_no_symmetries', optimal_supercell=False))]#,
# # (random_displace_atoms,
#  dict(noise_distribution='gaussian', displacement=0.2, displacement_scaled=0.10, create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
#       random_rotation=False, random_rotation_before=True, cell_type='standard_no_symmetries',
#       optimal_supercell=False))]  # ,
# (create_supercell_by_nb_atoms,
#  dict(min_nb_atoms=32, target_nb_atoms=256, random_rotation=True, cell_type='standard',
#       optimal_supercell=True)),
# (create_supercell_by_nb_atoms,
# dict(min_nb_atoms=32, target_nb_atoms=256, random_rotation=True, cell_type='standard',
#      optimal_supercell=True)),
# (create_supercell_by_nb_atoms,
#  dict(min_nb_atoms=32, target_nb_atoms=256, random_rotation=True, cell_type='standard',
#      optimal_supercell=True))]

# operation_names = ["_supercell_by_nb_atoms_min32_max256"]

# ending = "_symprecs_1e-06_angle_0.1.json.filter"
# accepted_keys = ['139', '141', '166', '194', '221', '225', '227', '229']


def get_key(string_to_split):
    # prototype name: RYvdvBLf1QdM5QJ_8DVve7CknkdK5_spgroup225_symprec_1E-6_angle_-1_.json.filter
    return string_to_split.rsplit('_', 5)[1].split('.', 1)[0].split('spgroup', 1)[1]


# spgroups_filters = group_filter_files(configs, ending, get_key)
# spgroups_jsons = get_paths_from_filter_dict(filter_dict=spgroups_filters, accepted_keys=accepted_keys)
#
# accepted_labels = [['139'], ['141'], ['166'], ['194'], ['221'], ['225'], ['227'], ['229']]

# =============================================================================
# Descriptor calculation
# =============================================================================
# json_list = []
# accepted_keys = ['139', '141', '166', '194', '221', '225', '227', '229']

# accepted_keys = ['166', '221']
# accepted_keys = ['221']

# target - 1e-9
# 139
# 141
# 166
# 194 - till 1e-7 ok, but from 1e-8 all 63 but we can use structures from e.g. 221
# 221 - ok
# 225 - ok
# 227 - ok
# 229 - ok

# for key in spgroups_jsons.keys():
#     if key in accepted_keys:
#         json_list += spgroups_jsons[key][:]

# json_list = []
# max_fold = len(data_folder)
# for idx in range(max_fold):
#     json_list.extend(
#         get_json_list(method='folder', drop_duplicates=False, data_folder=data_folder[idx], tmp_folder=tmp_folder)[:])

# # we should extract only the structures that have the same spacegroup number from 1e-01 to 1e-09
# ase_atoms_list = read_data(json_list, calc_spgroup=True, symprec=[1e-01, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09])
# ase_atoms_list = read_data(json_list, calc_spgroup=True, symprec=[1e-03, 1e-06, 1e-09])
# ase_db_file = write_ase_db(ase_atoms_list=ase_atoms_list, db_name='rh_bcc_sc_fcc_rh', main_folder=main_folder, folder_name='db_ase')
# sys.exit()
#
# ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms.db'

# accepted_labels = ['139', '141', '166', '194', '221', '225', '227', '229']
# accepted_labels = ['166', '194']
#
# ase_atoms_list_read = read_ase_db(db_path=ase_db_file)

# for label in accepted_labels:
#     logger.info('Calculating label: {}'.format(label))
#     filtered_ase_list = filter_ase_list_by_label(ase_atoms_list_read, main_folder=main_folder, filter_by=['spacegroup_nb'],
#                                                  accepted_labels=[label], folder_name='spg_' + str(label),
#                                                  write_to_file=True, symprec=[1e-03, 1e-06])
#
#     ase_db_file = write_ase_db(ase_atoms_list=filtered_ase_list,
#                                db_name='elemental_solids_ncomms_1e-3_1e-6_' + str(label), main_folder=main_folder,
#                                folder_name='db_ase')

# sys.exit(1)

# ase_db_files = ['/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_139.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_141.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_166.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_221.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_225.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_227.db',
# '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_229.db']
#
#
# ase_list_from_db = []
# for db in ase_db_files:
#     ase_list_from_db.extend(read_ase_db(db_path=db))
#
# logger.info("Len of ase list: {}".format(len(ase_list_from_db)))
# ase_db_file = write_ase_db(ase_atoms_list=ase_list_from_db,
#                                db_name='elemental_solids_ncomms_all_1e-3_1e-6_1e-9', main_folder=main_folder,
#                                folder_name='db_ase')

#   file_format='NOMAD' should be in write_json_for_nomad_sim
# json_list = write_json_for_nomad_sim(output_folder=tmp_json_folder, ase_atom_list=ase_atoms_list, label_name='nmd_checksum')

# spgroups = ['139', '141', '166', '221', '225', '227', '229']

# spgroups = ['227', '229']
# spgroups = ['166', '194']

# vac_ratios = [0.01]
#
# for vac_ratio in vac_ratios:
#     operations_on_structure_list = [
#     (create_vacancies,
#      dict(target_vacancy_ratio=vac_ratio, create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
#           random_rotation=False, random_rotation_before=True,
#           cell_type='standard_no_symmetries', optimal_supercell=False))]
#
#     for spgroup in spgroups:
#         logger.info("Calculating vacancy ratio {} for spgroup {}".format(vac_ratio, spgroup))
#
#         # ase_db_file = '/scratch/ziang/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_'+str(spgroup)+'.db'
#         ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_'+str(spgroup)+'.db'
#         ase_atoms_list = read_ase_db(db_path=ase_db_file)
#
#         desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list[:2],
#                                          tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
#                                          desc_file='spgroup_'+str(spgroup)+'_vac' + str(vac_ratio) + '.tar.gz', format_geometry='aims',
#                                          operations_on_structure=operations_on_structure_list,  # operations_on_structure=None,
#                                          nb_jobs=1, **kwargs)
#
#         # desc_file_path = '/scratch/ziang/2d_nature_comm/desc_folder/spgroup_' + str(spgroup) + '_vac' + str(vac_ratio) +'.tar.gz'
#         desc_file_path = '/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/spgroup_' + str(spgroup) + '_vac' + str(vac_ratio) +'.tar.gz'
#
#         target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)
#
#         df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='intensity',
#                                                  target_list=target_list,
#                                                  sprite_atlas_filename='descriptor_atlas_vac' + str(vac_ratio) + '_' + str(spgroup),
#                                                  configs=configs, normalize=True)
#
# sys.exit(1)

# ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_vac0.01.db'

# =============================================================================
# Load Descriptor file and Dataset preparation
# =============================================================================


spacegroups = ['139', '141', '166', '221', '225', '227', '229']

desc_file_path = []

vacs = ['0.01', '0.02', '0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7']

vacs = ['0.01']

# for vac in vacs:
#     logger.info("Vac ratio: {}".format(vac))
#     for spacegroup in spacegroups:
#         logger.info("Spgroup ratio: {}".format(spacegroup))
#
#         desc_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/desc_folder/spgroup_' + str(spacegroup) + \
#                     '_vac' + str(vac) + '.tar.gz'
#
#         desc_file_path.append(desc_file)
#
#     target_list, ase_atoms_list = load_descriptor(desc_files=desc_file_path, configs=configs)
#
#     ase_db_file = write_ase_db(ase_atoms_list=ase_atoms_list,
#                                db_name='elemental_solids_ncomms_1e-3_1e-6_1e-9_vac' + str(vac), main_folder=main_folder,
#                                folder_name='db_ase')
#
# sys.exit(1)

input_dims = (64, 64)

new_labels = {"bct_139": ["139"], "bct_141": ["141"], "hex/rh": ["166", "194"],
              #    "hex/rh": ["63", "69", "166", "191", "194"],
              "sc": ["221"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]}

dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets_2d')))
#
# path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(structure_list=ase_atoms_list,
#     target_list=target_list,
#     desc_metadata='intensity',
#     dataset_name='pristine_dataset',
#     target_name='spacegroup_nb_symprec_1e-09',
#     target_categorical=True,
#     input_dims=input_dims,
#     configs=configs,
#     dataset_folder=dataset_folder,
#     main_folder=main_folder,
#     desc_folder=desc_folder,
#     tmp_folder=tmp_folder,
#     disc_type=None, n_bins=None,
#     notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'. Spglib thresholds are 1e-3, 1e-6, 1e-9.",
#     new_labels=new_labels)

train_set_name = 'pristine_dataset'
path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_x.pkl')))
path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, train_set_name + '_y.pkl')))
path_to_summary_train = os.path.abspath(
    os.path.normpath(os.path.join(dataset_folder, train_set_name + '_summary.json')))

# test_set_name = 'pristine_dataset'
test_set_name = 'vac0.25_dataset'
path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_x.pkl')))
path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_y.pkl')))
path_to_summary_test = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, test_set_name + '_summary.json')))

x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                              path_to_summary=path_to_summary_train)

x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                           path_to_summary=path_to_summary_test)

params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
              "classes": dataset_info_train["data"][0]["classes"], # "checkpoint_filename": 'try_'+str(now.isoformat()),
              # "checkpoint_filename": 'ziletti_et_2018_rgb',  # the right one
              "checkpoint_filename": 'try1',  # the right one
              "batch_size": 32, "img_channels": 3}

text_labels = np.asarray(dataset_info_train["data"][0]["text_labels"])
numerical_labels = np.asarray(dataset_info_train["data"][0]["numerical_labels"])

text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

data_set_train = make_data_sets(x_train_val=x_train, y_train_val=y_train,
                                split_train_val=True, test_size=0.1, x_test=x_test, y_test=y_test,
                                stratified_splits=True)

# =============================================================================
# Neural network training and prediction
# =============================================================================

partial_model_architecture = partial(cnn_nature_comm_ziletti2018, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                     kernel_sizes=[7, 7, 7, 7, 7, 7], max_pool_strides=[2, 2], hidden_layer_size=128)

x_train = data_set_train.train.images
y_train = data_set_train.train.labels
x_val = data_set_train.val.images
y_val = data_set_train.val.labels
# x_test = data_set.test.images
# y_test = data_set.test.labels

# generate image of architecture
train_neural_network(x_train=x_train, x_val=x_val, y_train=y_train, y_val=y_val, configs=configs,
   partial_model_architecture=partial_model_architecture,
   checkpoint_dir=checkpoint_dir, neural_network_name=params_cnn["checkpoint_filename"], nb_epoch=5,
   training_log_file=training_log_file)

sys.exit(1)

data_set_predict = make_data_sets(x_train_val=x_train, y_train_val=y_train,
                                  split_train_val=False, x_test=x_test, y_test=y_test)

# load the data
x_test = data_set_predict.test.images
y_test = data_set_predict.test.labels

# select only 2% of the data to predict faster - can be omitted
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.01, random_state=0)
for train_index, test_index in sss.split(x_test, y_test):
    _, x_test_sub = x_test[train_index], x_test[test_index]
    _, y_test_sub = y_test[train_index], y_test[test_index]
    _, text_labels_sub = text_labels[train_index], text_labels[test_index]
    _, numerical_labels_sub = numerical_labels[train_index], numerical_labels[test_index]

predict_out = predict(x_test_sub, y_test_sub,
    configs=configs,
    batch_size=params_cnn["batch_size"],
    conf_matrix_file=conf_matrix_file,
    numerical_labels=numerical_labels_sub,
    text_labels=text_labels_sub,
    results_file=results_file)

target_pred_class = predict_out['target_pred_class']
target_pred_probs = predict_out['string_probs']
prob_predictions = predict_out['prob_predictions']
conf_matrix = predict_out['confusion_matrix']

# target_pred_class
sys.exit()
spacegroups = ['139', '141', '166', '221', '225', '227', '229']

y_true = []
y_spglib = []
y_nn = []

# spacegroups = ['221']
spacegroups = ['139', '141', '166', '221', '225', '227', '229']
symprec = 0.001
for spacegroup in spacegroups:
    ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_' + str(
        spacegroup) + '.db'
    ase_atoms_list = read_ase_db(db_path=ase_db_file)

    for ase_atom in ase_atoms_list[:10]:
        y_spglib.extend(get_spacegroup(ase_atom, symprec=symprec))
        y_true.append(spacegroup)

y_spglib = [str(item) for item in y_spglib]

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from ai4materials.utils.utils_plotting import plot_confusion_matrix

logger.info(accuracy_score(y_true, y_spglib))

conf_matrix = confusion_matrix(y_true, y_spglib)

plot_confusion_matrix(conf_matrix, classes=list(set(y_true)),
                      conf_matrix_file=conf_matrix_file,
                      normalize=False,
                      title='Confusion matrix',
                      title_pred_label='Spglib label (symprec='+str(symprec)+')',
                      cmap='Blues')


sys.exit(1)

desc_file_bcc_to_sc = ['descriptor_all_classes_8_samples.tar.gz',
                       #    'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_pristine.tar.gz',

                       ]

desc_file_bcc_to_amorphous = ['descriptor_all_classes_8_samples.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_pristine.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac05.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac10.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac15.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac20.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac25.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac30.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac35.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac40.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac45.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac50.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac55.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac60.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac65.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac70.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac75.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac80.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac85.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac90.tar.gz',
                              'diam_to_amorphous_spg227_supercell_by_nb_atoms_min32_max256_vac95.tar.gz']

desc_file_fcc_to_amorphous_vac = ['descriptor_all_classes_8_samples.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_pristine.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac05.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac10.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac15.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac20.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac25.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac30.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac35.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac40.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac45.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac50.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac55.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac60.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac65.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac70.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac75.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac80.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac85.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac90.tar.gz',
                                  'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_vac95.tar.gz']

desc_file_sc_to_amorphous_vac = ['descriptor_all_classes_8_samples.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_pristine.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac05.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac10.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac15.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac20.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac25.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac30.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac35.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac40.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac45.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac50.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac55.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac60.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac65.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac70.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac75.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac80.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac85.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac90.tar.gz',
                                 'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_vac95.tar.gz']

desc_file_bcc_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                   'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz'
                                   # ,
                                   #    'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_gaussian_scaled_35.tar.gz'
                                   ]

desc_file_sc_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz',
                                  'sc_to_amorphous_spg221_supercell_by_nb_atoms_min32_max256_gaussian_scaled_35.tar.gz']

desc_file_fcc_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_025.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_075.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_125.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_175.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_225.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_275.tar.gz',
                                   'fcc_to_amorphous_spg225_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz']

desc_file_hex_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_025.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_075.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_125.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_175.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_225.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                   'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_275.tar.gz']
#    'hex_to_amorphous_spg194_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz']     

desc_file_rh_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256.tar.gz',
                                  #    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_025.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                  #    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_075.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                  #    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_125.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                  #    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_175.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                  #    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_225.tar.gz',
                                  'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz']  # ,
#    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_275.tar.gz',
#    'rh_to_amorphous_spg166_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz']     


desc_file_bct141_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                      'bct_to_amorphous_spg141_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz']

desc_file_bct139_to_amorphous_disp = ['descriptor_all_classes_8_samples.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_05.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_10.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_15.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_20.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_25.tar.gz',
                                      'bct_to_amorphous_spg139_supercell_by_nb_atoms_min32_max256_gaussian_scaled_30.tar.gz']

desc_file_bcc_to_amorphous_vac = ['descriptor_all_classes_8_samples.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_pristine.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac05.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac10.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac15.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac20.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac25.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac30.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac35.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac40.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac45.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac50.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac55.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac60.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac65.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac70.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac75.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac80.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac85.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac90.tar.gz',
                                  'bcc_to_amorphous_spg229_supercell_by_nb_atoms_min32_max256_vac95.tar.gz']

desc_file_bcc_to_sc = ["descriptor_all_classes_8_samples.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_all_bcc.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio05.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio10.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio15.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio20.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio25.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio30.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio35.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio40.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio45.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio50.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio55.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio60.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio65.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio70.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio75.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio80.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio85.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio90.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio95.tar.gz",
                       "bcc_to_sc229_supercell_by_nb_atoms_min32_max256_ratio100.tar.gz"]

rh_bcc_sc_fcc_rh_list = []
prefix_file = "rh_bcc_sc_fcc_"
suffix_file = "_fig4_paper_nomad.json.tar.gz"
for i in range(0, 161):
    rh_bcc_sc_fcc_rh_list.append(prefix_file + str(i) + suffix_file)

# print rh_bcc_sc_fcc_rh_list
desc_file_rh_bcc_sc_fcc_rh = ['descriptor_all_classes_8_samples.tar.gz']
desc_file_rh_bcc_sc_fcc_rh.extend(rh_bcc_sc_fcc_rh_list)

desc_file_bct_bcc_fcc = ['descriptor_all_classes_8_samples.tar.gz', 'bct_bcc_fcc_0_nomad.json.tar.gz',
                         'bct_bcc_fcc_1_nomad.json.tar.gz', 'bct_bcc_fcc_2_nomad.json.tar.gz',
                         'bct_bcc_fcc_3_nomad.json.tar.gz', 'bct_bcc_fcc_4_nomad.json.tar.gz',
                         'bct_bcc_fcc_5_nomad.json.tar.gz', 'bct_bcc_fcc_6_nomad.json.tar.gz',
                         'bct_bcc_fcc_7_nomad.json.tar.gz', 'bct_bcc_fcc_8_nomad.json.tar.gz',
                         'bct_bcc_fcc_9_nomad.json.tar.gz', 'bct_bcc_fcc_10_nomad.json.tar.gz']

desc_file_bct_diam = ['descriptor_all_classes_8_samples.tar.gz', 'bct_diam_0_nomad.json.tar.gz',
                      'bct_diam_1_nomad.json.tar.gz', 'bct_diam_2_nomad.json.tar.gz', 'bct_diam_3_nomad.json.tar.gz',
                      'bct_diam_4_nomad.json.tar.gz', 'bct_diam_5_nomad.json.tar.gz', 'bct_diam_6_nomad.json.tar.gz',
                      'bct_diam_7_nomad.json.tar.gz', 'bct_diam_8_nomad.json.tar.gz', 'bct_diam_9_nomad.json.tar.gz',
                      'bct_diam_10_nomad.json.tar.gz']

# for desc_file in desc_file_rh_bcc_sc_fcc_rh:
#    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))
# #
# df_filepath = generate_facets_input(desc_folder=desc_folder, main_folder=main_folder,
#    input_dims=input_dims, desc_file_list=desc_file_rh_bcc_sc_fcc_rh, tmp_folder=tmp_folder)
#
# sys.exit(1)

# desc_file_list_train = found_operations['_pristine']
# desc_file_list_train = ['descriptor_all_classes_8_samples.tar.gz']
#
# path_to_x_train, path_to_y_train, path_to_summary_train = prepare_dataset(desc_file_list=desc_file_list_train,
#    target_name='spacegroup_number_0.001_1.0', target_categorical=True,
# #    target_name='spacegroup_number_actual_0.001_1.0', target_categorical=True,
#    dataset_dir=dataset_dir,
#    dataset_name='small_pristine',
#    input_dims=input_dims,
#    main_folder=main_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder,
#    disc_type=None, n_bins=None,
#    split_train_val=False,
#    notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'.",
#    new_labels=new_labels
#    )
##
##
# desc_file_list_test = ['descriptor_all_classes_8_samples.tar.gz'] + found_operations['_vac'][:5]

desc_file_list_test = desc_file_rh_bcc_sc_fcc_rh

# desc_file_list_test = [
#    'descriptor_try1_cut.tar.gz'] 

# desc_file_list_test = desc_file_fcc_to_amorphous
# desc_file_list_test = desc_file_sc_to_amorphous
# print "len(desc_file_list_test)", len(desc_file_list_test)
# desc_file_list_test = desc_file_rh_to_fcc_to_sc_to_bcc


# path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(desc_file_list=desc_file_bct139_to_amorphous_disp,
#    target_name='spacegroup_number_0.001_1.0', target_categorical=True,
#    dataset_dir=dataset_dir,
#    dataset_name='bct139_to_amorphous_disp',
#    input_dims=input_dims,
#    main_folder=main_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder,
#    disc_type=None, n_bins=None,
#    split_train_val=False,
#    notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'. Added 1 sample for each class.",
#    new_labels=new_labels
#    )

# path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(desc_file_list=desc_file_rh_bcc_sc_fcc_rh,
#    target_name='spacegroup_number_0.001_1.0', target_categorical=True,
#    dataset_dir=dataset_dir,
#    dataset_name='rh_bcc_sc_fcc_rh_new',
#    input_dims=input_dims,
#    main_folder=main_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder,
#    disc_type=None, n_bins=None,
#    split_train_val=False,
#    notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'. Added 1 sample for each class.",
#    new_labels=new_labels
#    )
#

# path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(desc_file_list=desc_file_bcc_to_sc,
#    target_name='spacegroup_number_0.001_1.0', target_categorical=True,
#    dataset_dir=dataset_dir,
#    dataset_name='bcc_to_sc229_supercell_256',
#    input_dims=input_dims,
#    main_folder=main_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder,
#    disc_type=None, n_bins=None,
#    split_train_val=False,
#    notes="Incremented by 2 the atomic number. 166, and 194 merged in 'hex/rh'. Added 1 sample for each class.",
#    new_labels=new_labels
#    )
#

# path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(desc_file_list=desc_file_rh_bcc_sc_fcc_rh,
#    target_name='spacegroup_number_0.001_1.0', target_categorical=True,
#    dataset_dir=dataset_dir,
#    dataset_name='rh_bcc_sc_fcc_rh_new',
#    input_dims=input_dims,
#    main_folder=main_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder,
#    disc_type=None, n_bins=None,
#    split_train_val=False,
#    notes="Incremented by 2 the atomic number. Transition rh-bcc-sc-fcc. Added 1 sample for each class.",
#    new_labels=new_labels
#    )

train_set_name = 'supercell_by_nb_atoms_min32_max256'
# train_set_name = 'rh_to_amorphous_disp'
path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, train_set_name + '_x.pkl')))
path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, train_set_name + '_y.pkl')))
path_to_summary_train = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, train_set_name + '_summary.json')))
#
# test_set_name = 'rh_bcc_sc_fcc_rh_new'
# path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_x.pkl')))
# path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_y.pkl')))
# path_to_summary_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_summary.json')))

# test_set_name = 'rh_bcc_sc_fcc_rh'
# test_set_name = 'rh_bcc_sc_fcc_rh_new'
test_set_name = 'bcc_to_sc229_supercell_256'
# test_set_name = 'bcc_to_amorphous_disp'
# test_set_name = 'sc_to_amorphous_disp'
# test_set_name = 'fcc_to_amorphous_disp'
# test_set_name = 'bct141_to_amorphous_disp'
# test_set_name = 'bct139_to_amorphous_disp'
# test_set_name = 'bcc_to_amorphous_disp'
# test_set_name = 'rh_to_amorphous_disp'
# test_set_name = 'bct_to_bcc_to_fcc'
# test_set_name = 'bct_diam'

# test_set_name = 'supercell_by_nb_atoms_min32_max256_disp008'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_disp010'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac25'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac01'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac02'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac05'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac10'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac15'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac30'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac40'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac50'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac60'
# test_set_name = 'supercell_by_nb_atoms_min32_max256_vac70'

path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_x.pkl')))
path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_y.pkl')))
path_to_summary_test = os.path.abspath(os.path.normpath(os.path.join(dataset_dir, test_set_name + '_summary.json')))

x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                              path_to_summary=path_to_summary_train,
                                                              input_dims=input_dims)

x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                           path_to_summary=path_to_summary_test, input_dims=input_dims)

params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
              "classes": dataset_info_train["data"][0]["classes"], # "checkpoint_filename": 'try_'+str(now.isoformat()),
              "checkpoint_filename": 'ziletti_et_2018_rgb',  # the right one
              "batch_size": 32, "img_channels": 3}

text_labels = np.asarray(dataset_info_train["data"][0]["text_labels"])
numerical_labels = np.asarray(dataset_info_train["data"][0]["numerical_labels"])

text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])
#
#
data_set_train = make_data_sets(input_dims=input_dims, train_val_images=x_train, train_val_labels=y_train,
                                split_train_val=True, test_size=0.1, test_images=x_test, test_labels=y_test,
                                flatten_images=False)

# =============================================================================
# Neural network training and prediction
# =============================================================================

partial_model_architecture = partial(model_deep_cnn_struct_recognition, conv2d_filters=[32, 32, 16, 16, 8, 8],
                                     kernel_sizes=[7, 7, 7, 7, 7, 7], max_pool_strides=[2, 2], hidden_layer_size=128)

##
# train_cnn_keras(
#    data_set=data_set_train,
#    nb_classes=params_cnn["nb_classes"],   
#    input_dims=input_dims,
#    partial_model_architecture=partial_model_architecture,
#    batch_size=params_cnn["batch_size"],
#    img_channels=3,
#    checkpoint_dir=checkpoint_dir,
#    checkpoint_filename=params_cnn["checkpoint_filename"],
#    nb_epoch=5, 
#    training_log_file=training_log_file,
#    early_stopping=False,
#    data_augmentation=False)
#                      
# sys.exit(1)
##
# data_set_predict = make_data_sets(input_dims=input_dims,
#     train_val_images=x_train,
#     train_val_labels=y_train,
#     split_train_val=False,
#     test_size=0.1,
#     test_images=x_test,
#     test_labels=y_test,
#     flatten_images=False)
# ##
# target_pred_class, target_pred_probs, prob_predictions, conf_matrix = predict_cnn_keras(
#     data_set_predict, params_cnn["nb_classes"], input_dims,
#     img_channels=3, batch_size=params_cnn["batch_size"],
#     checkpoint_dir=checkpoint_dir,
#     checkpoint_filename=params_cnn["checkpoint_filename"],
#     show_model_acc=True,
#     predict_probabilities=True,
#     plot_conf_matrix=True,
#     conf_matrix_file=conf_matrix_file,
#     numerical_labels=numerical_labels,
#     text_labels=text_labels,
#     results_file=results_file)

#    
print "Execution time: ", datetime.now() - startTime

# sys.exit(1)


### plot neural network training log
# plot_save_cnn_results(training_log, accuracy=True, cross_entropy_loss=True,
#   show_plot=True)


# ==============================================================================
# Structural transitions
# ==============================================================================

palette = ['indigo', 'saddlebrown', 'black', 'green', 'blue', 'gold', 'red']
labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"]
#
## bcc --> amorphous
# df_results = aggregate_struct_trans_data(results_file_bcc_to_amorphous,
#    nb_rows_to_cut=8,
#    nb_samples=1,
#    nb_order_param_steps=11, min_order_param=0.5, max_order_param=1.5, 
#    # extract all classes
#    prob_idxs=range(params_cnn["nb_classes"]))
#
# print "classes", params_cnn["classes"]
#
# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    palette=palette,
#    labels=labels,
#    nb_order_param_steps=11,     
#    filename_suffix=".svg", 
#    title="Structural transition: bct to bcc tp fcc ", 
#    x_label="q", show_plot=True)
#
# sys.exit(1)
##
#
#
# bcc --> sc
df_results = aggregate_struct_trans_data(results_file, nb_rows_to_cut=8, nb_samples=100, nb_order_param_steps=21,
                                         min_order_param=0.0, max_order_param=1.0,  # extract all classes
                                         prob_idxs=range(params_cnn["nb_classes"]))

print "classes", params_cnn["classes"]
print df_results.head()

make_crossover_plot(df_results, results_file, prob_idxs=[0, 6], palette=palette, labels=labels, nb_order_param_steps=21,
                    filename_suffix=".svg", title="Structural transition: bcc to sc", x_label="q", show_plot=True)

sys.exit(1)

#
#
# rh -> bcc -> sc -> fcc -> rh
# df_results = aggregate_struct_trans_data(results_file,
#    nb_rows_to_cut=8,
#    nb_samples=1,
#    nb_order_param_steps=161, min_order_param=1.0, max_order_param=5.0,
#    # extract all classes
#    prob_idxs=range(params_cnn["nb_classes"]))
#
# print "classes", params_cnn["classes"]
#
# make_crossover_plot(df_results, results_file,
#    prob_idxs=[0, 4, 5, 6],
#    palette=palette,
#    labels=labels,
#    nb_order_param_steps=161,
#    max_nb_ticks=17,
#    filename_suffix=".svg",
#    title="Structural transition: bcc->rh->sc->rh->fcc->rh",
#    x_label="q", show_plot=False)
#
# sys.exit(1)
#


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

# df_results = aggregate_struct_trans_data(results_file_bcc_to_amorphous,
#    nb_rows_to_cut=8,
#    nb_samples=1,
#    nb_order_param_steps=16, max_order_param=1.00, 
#    # extract all classes
#    prob_idxs=range(params_cnn["nb_classes"]))


# df_results = aggregate_struct_trans_data(results_file_bcc_to_amorphous,
#    nb_rows_to_cut=8,
#    nb_samples=10,
#    nb_order_param_steps=20, max_order_param=0.95, 
#    # extract all classes
#    prob_idxs=range(params_cnn["nb_classes"]))

# print "classes", params_cnn["classes"]
# classes [u'bcc', u'bct_139', u'bct_141', u'diam', u'fcc', u'hex/rh', u'sc']

# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"],
#    nb_order_param_steps=20,     
#    filename_suffix=".png", 
#    title="From body-centered-cubic (bcc) to amorphous", 
#    x_label="Vacancies (atoms removed) [%]", show_plot=False)

# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"],
#    nb_order_param_steps=20,     
#    filename_suffix=".png", 
#    title="From face-centered-cubic (fcc) to amorphous", 
#    x_label="Vacancies (atoms removed) [%]", show_plot=False)

# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"],
#    nb_order_param_steps=20,     
#    filename_suffix=".png", 
#    title="From simple-cubic (fcc) to amorphous", 
#    x_label="Vacancies (atoms removed) [%]", show_plot=False)

# make_crossover_plot(df_results, results_file_bcc_to_amorphous,
#    prob_idxs=[0, 1, 2, 3, 4, 5, 6], 
#    labels = ["$p_{bcc}$", "$p_{bct_{139}}$", "$p_{bct_{141}}$", "$p_{diam}$", "$p_{fcc}$", "$p_{hex/rh}$", "$p_{sc}$"],
#    nb_order_param_steps=16,     
#    filename_suffix=".png", 
#    title="From face-centered-cubic (fcc) to amorphous", 
#    x_label="Vacancies (atoms removed) [%]", show_plot=False)

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


desc_file_list = ['descriptor_for_filter.tar.gz']

#
for desc_file in desc_file_list:
    #
    ##    desc_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'descriptor_rnh_all.tar.gz')))
    #
    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))
    #
    desc_images = extract_images(filename=desc_file_path, filetype='descriptor_files', input_dims=input_dims,
                                 desc_folder=desc_folder, tmp_folder=tmp_folder)
##
#    
images = desc_images
model_weights_file = os.path.abspath(
    os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] + '.h5')))
model_arch_file = os.path.abspath(
    os.path.normpath(os.path.join(checkpoint_dir, params_cnn["checkpoint_filename"] + '.json')))
#

print "images.shape", images.shape

plot_att_response_maps(images, model_arch_file, model_weights_file, figure_dir, nb_conv_layers=6, nb_top_feat_maps=8,
                       layer_nb='all',  # layer_nb=[0],
                       plot_all_filters=True, plot_filter_sum=True, plot_summary=True)
#
#

print "Execution time: ", datetime.now() - startTime

sys.exit(1)

# read panda dataframe and plot results
# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv')))
# plot_bcc_to_scc(results_file, show_plot=True)

# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_diamond_fcc.csv')))
# plot_diamond_to_fcc(results_file, show_plot=True)

# results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_bcc_disorder_all.csv')))
# plot_bcc_disorder(results_file, show_plot=True)

print "Execution time: ", datetime.now() - startTime

# sys.exit(1)


target_pred_list = target_pred_probs
target_list = num_labels

# embed_params={'learning_rate': 500}
#
# model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'try_all_1.h5')))
# model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'try_all_1.json')))
#
##model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7.h5')))
##model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7.json')))
#
## embedding is not compatible with desc_file being a list
##calc_embedding(embed_method='pca', desc_type='diffraction_2d',
# calc_embedding(embed_method='tsne_pca', desc_type='diffraction_2d',
##calc_embedding(embed_method='tsne', desc_type='diffraction_2d',
##calc_embedding(embed_method='spectral_embedding', desc_type='diffraction_2d',
##calc_embedding(embed_method='mds', desc_type='diffraction_2d',
##    lookup_file=lookup_file, desc_file=desc_file_list_test[0],
#    target_categorical=True,
#    input_dims=input_dims,
#    lookup_file=lookup_file, desc_file=desc_file_list_test,
#    #desc_folder=tmp_folder, 
#    desc_folder=desc_folder,
#    standardize='True',
#    target_name='target', embed_params=embed_params, 
#    use_xray_img=True,
#    model_arch_file=model_arch_file, 
#    model_weights_file=model_weights_file,
#    nb_nn_layer=-1,
#    path_to_x_test=path_to_x_test
#    )
#    
#
# json_list, frame_list, x_list, y_list, foo = get_json_list(method='file', data_folder=data_folder,
#        #path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
#        path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)


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
    for j in range(len(operations_on_structure)):
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
le.fit(class_labels)
target_class = le.transform(class_labels)
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

for desc_file_list_test_ in desc_file_list_test:
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


plot_misclassified_only = True

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
                descriptor=descriptor, operations_on_structure=operations_on_structure, xray_img_list=xray_img_list,
                file_format='NOMAD', clustering_x_list=x_list, clustering_y_list=y_list, target_list=target_list,
                is_classification=True, target_class_names=classes, target_pred_list=target_pred_list, target_unit='',
                clustering_point_size=12, tmp_folder=tmp_folder, control_file=control_file, cell_type=cell_type,
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
##        calc_descriptor(desc_type='diffraction_2d', file_format='NOMAD',
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
# start_folder=0
# for i in range(start_folder, len(data_folder)):
#
#
#     for accepted_label in accepted_labels:
#         json_list = []
#
#         json_list.append(get_json_list(method='folder', drop_duplicates=False,
#            data_folder=data_folder[i], tmp_folder=tmp_folder))
#
#         json_list = [item for sublist in json_list for item in sublist]
#         # json_list = json_list[:10]
#
#         # filtering the json_list
#         json_list = filter_json_list(file_format='NOMAD',
#         json_list=json_list, tmp_folder=tmp_folder,
#         desc_folder=desc_folder,
#         #desc_folder=tmp_folder,
#         filter_by=['spacegroup_number'],
#         symprecs=symprecs,
#         angle_tolerances=angle_tolerances,
#         cell_type='standard',
#         accepted_labels=accepted_label,
#         write_to_file=True,
#         filtered_file=desc_names[i] + '_spgroup' + str(accepted_label[0]) + '_symprecs_' + str(symprecs[0]) + '_angle_' + str(angle_tolerances[0]) + '.json.filter',
#         operations_on_structure=operations_on_structure_list[0],
#         **kwargs)
#
# sys.exit(1)
