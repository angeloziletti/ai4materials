#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/10/16"


# hack to change to local/Beaker mode in all files in the packages
# DEPRECATED
#import __builtin__
#__builtin__.isBeaker = False

import sys, os, os.path

base_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.normpath(os.path.join(base_dir,"../../../python-common/common/python"))
nomad_sim_dir = os.path.normpath(os.path.join(base_dir,"../python-modules/"))
atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data')) 
#visualization_dir = os.path.normpath(os.path.join(base_dir,"../python-modules/visualization/"))

if not common_dir in sys.path:
    sys.path.insert(0, common_dir) 
    sys.path.insert(0, nomad_sim_dir) 
    sys.path.insert(0, atomic_data_dir) 
#    sys.path.insert(0, visualization_dir) 
    
# hack to get it running locally - TO BE REMOVED
#sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'python-modules'))

#print sys.path


import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

from nomad_sim.wrappers import get_json_list, calc_descriptor 
from nomad_sim.utils_crystals import create_supercell
from nomad_sim.utils_crystals import create_supercell_by_nb_atoms
from nomad_sim.utils_crystals import create_vacancies
from nomad_sim.utils_crystals import random_displace_atoms
from nomad_sim.utils_crystals import substitute_atoms
from nomad_sim.utils_crystals import filter_json_list
from nomad_sim.utils_plotting import aggregate_struct_trans_data, make_crossover_plot
from nomad_sim.utils_plotting import make_multiple_image_plot, plot_save_cnn_results
from nomad_sim.utils_data_retrieval import extract_img_list, create_sprite_atlas
from nomad_sim.utils_data_retrieval import extract_images, generate_facets_input, write_summary_file
from nomad_sim.descriptors import XrayDiffraction
from nomad_sim.model_cnn import run_cnn_model
from nomad_sim.cnn import model_deep_cnn_struct_recognition
from nomad_sim.cnn import model_shallow_cnn_struct_recognition
from nomad_sim.deconv_resp_maps import plot_att_response_maps



'''Note: to use the tutorial on the Beaker Notebook:
1) tmp_folder = '/home/beaker/.beaker/v1/web/tmp/'
2) uncomment the lines "HTML(filename)"
3) control_file = '/home/beaker/.beaker/v1/web/tmp/control.json'
'''


data_folder=[
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Rnh_4DFTJQgTSOib4e4d-5GByiTVB',
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/R10ncY1AZG6X9y-Nj8F0_DiN8NeLD', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RsLoZhSAdK0BopfI2T4B5pLfMyjVN', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RMGpPc3B_HiR0D-oLE4ND66HmYdH-', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Re2mnhOAs6ZNqvTY1p-W2RavinjOM', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/R9usAWjw2xq9F8zW-66jyCyeDLlDa', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RkxmUCgPxt-9xDdIpr5xqPQK8PC9H', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RdzeezGR0W5wGEpGYEqOq7AygYS9J', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Rc_XxYadb0ZlfBVLqCNo-EtVocxv8', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RKXqE9xPCiLlufNK0n4pbtzdbID5H',
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RYvdvBLf1QdM5QJ_8DVve7CknkdK5', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RUt3qcReY6SJO6fIJ5jangTSlMjaQ', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/REg0D-KojGnrw51EYl2Q2rCYOIfJM', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RSkoltrNkpZwp1xpi_Oj1jO4IndC5', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RA-tqhSlH5idfPW_3UxE80I7BBL6s', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RayT1o-XjyZaWdlVS_Fk8nssdO1w9', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g', 
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r',
    '/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Ra8nAuJOgxGwSytw1scU5BTeB3ozo'] 

# data folder needs to be a list
#data_folder = [data_folder[0]]
#data_folder = data_folder[0:2]



# define folders
main_folder = '/home/ziletti/Documents/calc_xray/faces_crystals_tutorial/'
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp'))) 
example_data_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'example-data'))) 
desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder'))) 
checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models'))) 
figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps'))) 
example_data_bcc_to_sc_folder = os.path.abspath(os.path.normpath(os.path.join(example_data_folder, 'bcc_to_sc'))) 
example_data_diam_to_fcc_folder = os.path.abspath(os.path.normpath(os.path.join(example_data_folder, 'diam_to_fcc'))) 
example_data_bcc_to_amorphous_folder = os.path.abspath(os.path.normpath(os.path.join(example_data_folder, 'bcc_to_amorphous'))) 


# define files 
desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder,'desc_info.json.info'))) 
desc_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, 'descriptor.tar.gz')))
lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'lookup.dat'))) 
control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'control.json'))) 
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results.csv'))) 
results_file_bcc_to_sc = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv'))) 
results_file_diam_to_fcc = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_diam_to_fcc.csv'))) 
results_file_bcc_to_amorphous = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_to_amorphous.csv'))) 
conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'confusion_matrix.png'))) 

checkpoint_filename='pristine_all_kernel_7_final'

path_to_x_train = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_xtrain.pkl'))) 
path_to_y_train = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_ytrain.pkl'))) 
path_to_x_val = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_xval.pkl'))) 
path_to_y_val = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_yval.pkl'))) 
path_to_x_test = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_xtest.pkl'))) 
path_to_y_test = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'data_set_ytest.pkl'))) 
model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'pristine_all_kernel_7_final.h5')))
model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, 'pristine_all_kernel_7_final.json')))
training_log = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir,'training.log'))) 


# read list from files
fm3m = []
fd3m = []
im3m = []
pm3m = []

for root, dirs, files in os.walk(example_data_folder, topdown=True):
    for file_ in files:
        if file_.endswith("Fm-3m.json.filter"):
            fm3m.append(os.path.join(root, file_))
        if file_.endswith("Fd-3m.json.filter"):
            fd3m.append(os.path.join(root, file_))
        if file_.endswith("Im-3m.json.filter"):
            im3m.append(os.path.join(root, file_))
        if file_.endswith("Pm-3m.json.filter"):
            pm3m.append(os.path.join(root, file_))

fm3m_json = []
fd3m_json = []
im3m_json = []
pm3m_json = []

for in_file in fm3m:
    with open(in_file) as json_file:
        data = json.load(json_file)
        fm3m_json.append(data["data"][0]["filtered_json_list"])

for in_file in fd3m:
    with open(in_file) as json_file:
        data = json.load(json_file)
        fd3m_json.append(data["data"][0]["filtered_json_list"])
        
for in_file in im3m:
    with open(in_file) as json_file:
        data = json.load(json_file)
        im3m_json.append(data["data"][0]["filtered_json_list"])

for in_file in pm3m:
    with open(in_file) as json_file:
        data = json.load(json_file)
        pm3m_json.append(data["data"][0]["filtered_json_list"])
        
# flatten the lists
fm3m_json = [item for sublist in fm3m_json for item in sublist]
fd3m_json = [item for sublist in fd3m_json for item in sublist]
im3m_json = [item for sublist in im3m_json for item in sublist]
pm3m_json = [item for sublist in pm3m_json for item in sublist]

print len(fm3m_json)
print len(fd3m_json)
print len(im3m_json)
print len(pm3m_json)
    
#==============================================================================
# Descriptor calculation
#==============================================================================

user_param_source = {
    'wavelength': 5.0E-12,     
    'pulse_energy': 1E-6,  
    'focus_diameter': 1E-6 
    }

user_param_detector = {  
    'distance': 0.1,       
    'pixel_size': 4E-4,    
    'nx': 64,
    'ny': 64
}

input_dims = (64, 64)
angles_d = [45.0]

kwargs = {'ndim': 2,
          'desc_space': 'k-space', 
          'user_param_source': user_param_source,
          'user_param_detector': user_param_detector,
          'angles_d': angles_d,
          'rotation': True,
          'atoms_scaling': 'avg_distance_nn'}

cell_type = 'standard'

operations_on_structure = [
#    (create_supercell, {'replicas': [2, 2, 2]}),
#    (create_supercell, {'replicas': [3, 3, 3]}),
#    (random_displace_atoms, {'displacement': 0.05, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),
#    (random_displace_atoms, {'displacement': 0.10, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),
#    (create_vacancies, {'target_vacancy_ratio': 0.10, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),
#    (create_vacancies, {'target_vacancy_ratio': 0.30, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 128}),
#    (substitute_atoms, {'target_sub_ratio': 0.50, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64, 'max_n_sub_species': 1}),
]
    
 
descriptor = XrayDiffraction(**kwargs)

json_list = fm3m_json[:1] + fd3m_json[:1] + im3m_json[:1] + pm3m_json[:1]


#calc_descriptor(desc_type='xray', file_format='NOMAD',
#    json_list=json_list, tmp_folder=tmp_folder,
#    desc_folder=tmp_folder,
##    desc_file='descriptor.tar.gz',
#    desc_file='descriptor_all_classes_4_samples.tar.gz',
#    desc_info_file=desc_info_file,
#    grayscale=True,
#    target_list=np.zeros(len(json_list)), 
#    cell_type=cell_type,
#    operations_on_structure=operations_on_structure,
#    **kwargs) 
#
#sys.exit(1)

#geometry_images = extract_images(filename=desc_file, 
#    filetype='geometry_thumbnails',
#    input_dims=input_dims, desc_folder=tmp_folder, tmp_folder=tmp_folder)
#    
#geometry_images = np.reshape(geometry_images, (-1, input_dims[0], input_dims[1]))
#
#geo_sprite_atlas_file = create_sprite_atlas(images=geometry_images, main_folder=main_folder,
#    sprite_atlas_filename="geometries_atlas", max_imgs_row=len(operations_on_structure))
#    
#  
#desc_images = extract_images(filename=desc_file, 
#    filetype='2d_diffraction_images_ks',
#    input_dims=input_dims, desc_folder=tmp_folder, tmp_folder=tmp_folder)
#desc_images = np.reshape(desc_images, (-1, input_dims[0], input_dims[1]))
#
#desc_sprite_atlas_file = create_sprite_atlas(images=desc_images, main_folder=main_folder,
#    sprite_atlas_filename="descriptors_atlas", max_imgs_row=len(operations_on_structure))

desc_file_list_train = [
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128.tar.gz']

#
#desc_file_list_test = [
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128.tar.gz'

desc_file_list_test_vac10 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_vacancy_10.tar.gz']
    

desc_file_list_test_vac20 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_vacancy_20.tar.gz']
    
desc_file_list_test_vac30 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_vacancy_30.tar.gz']
    
desc_file_list_test_vac40 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_vacancy_40.tar.gz']

desc_file_list_test_vac50 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz']

desc_file_list_disp05 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.05A.tar.gz']    

desc_file_list_disp10 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.10A.tar.gz']   
    
desc_file_list_disp15 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz']   

desc_file_list_disp20 = [
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'R9usAWjw2xq9F8zW-66jyCyeDLlDa_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RA-tqhSlH5idfPW_3UxE80I7BBL6s_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RayT1o-XjyZaWdlVS_Fk8nssdO1w9_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RcC8TDWGWCtQLhWeB2a1N8y9Q7y4r_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'Rc_XxYadb0ZlfBVLqCNo-EtVocxv8_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RdzeezGR0W5wGEpGYEqOq7AygYS9J_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'Re2mnhOAs6ZNqvTY1p-W2RavinjOM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'REg0D-KojGnrw51EYl2Q2rCYOIfJM_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RkxmUCgPxt-9xDdIpr5xqPQK8PC9H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RKXqE9xPCiLlufNK0n4pbtzdbID5H_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RLbbgx7klbZ7ZdO5O_YABQGjBOZ9g_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RSkoltrNkpZwp1xpi_Oj1jO4IndC5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RUt3qcReY6SJO6fIJ5jangTSlMjaQ_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz',
    'RYvdvBLf1QdM5QJ_8DVve7CknkdK5_supercell_by_nb_atoms_min32_max128_gauss_disp_0.20A.tar.gz']   


    
#desc_file_list_train = [desc_file_list_train[0]]
#desc_file_list_test = [desc_file_list_test[0]]


#df_filepath = generate_facets_input(desc_folder=desc_folder, input_dims=input_dims, 
#        desc_file_list=desc_file_list, main_folder=main_folder, tmp_folder=tmp_folder)


#desc_file_list_train = desc_file_list_train[:1]
#desc_file_list_test = desc_file_list_test[:1]

#desc_file_list_train = ['descriptor.tar.gz']
#desc_file_list_test = ['descriptor.tar.gz']

#target_pred_class, target_pred_probs, num_labels, class_labels = 


#checkpoint_filename = 'test_shallow'

#partial_model_architecture = partial(model_deep_cnn_struct_recognition,
#        conv2d_filters=[32, 16, 8, 8, 16, 32],
#        kernel_sizes=[7, 7, 7, 7, 7, 7], 
#        max_pool_strides=[2, 2],
#        hidden_layer_size=128)

partial_model_architecture = partial(model_deep_cnn_struct_recognition,
        conv2d_filters=[32, 16, 12, 12, 8, 8],
        kernel_sizes=[7, 7, 7, 7, 7, 7], 
        max_pool_strides=[2, 2],
        hidden_layer_size=128)
#        
#        conv2d_filters=[32, 16, 4, 4, 16, 32],
# with 10 epoch and Rnh[0] this is the best so far

#checkpoint_filename='my_convnet_all_10_epochs'

#checkpoint_filename='pristine_all_kernel_7_e'

#checkpoint_filename = 'pristine_all_kernel_7_final'
#checkpoint_filename='pristine_all_kernel_7_epoch_5'

# the right one
#checkpoint_filename='pristine_all_kernel_7_final_from_all_pristine'

#checkpoint_filename='convnet_ziletti_et_al_2017_epoch3'
#checkpoint_filename='convnet_ziletti_et_al_2017'
checkpoint_filename='convnet_ziletti_et_al_2017_epoch2'


startTime = datetime.now()

        
#run_cnn_model(method='xray_cnn', 
#    desc_type='xray',
#    train=True, 
#    split_train_val=True,    
#    read_from_file=False,
#    partial_model_architecture=partial_model_architecture,
#    path_to_x_train=path_to_x_train,
#    path_to_y_train=path_to_y_train,
#    path_to_x_val=path_to_x_val,
#    path_to_y_val=path_to_y_val,
#    path_to_x_test=path_to_x_test,
#    path_to_y_test=path_to_y_test,
#    target_name='spacegroup_symbol', target_categorical=True,
#    desc_file_list_train=desc_file_list_train,
#    desc_file_list_test=desc_file_list_train,
#    input_dims=input_dims,
#    desc_folder=desc_folder,
##    desc_folder=tmp_folder,
#    tmp_folder=tmp_folder, 
#    lookup_file=lookup_file, 
#    control_file=control_file,
#    results_file=results_file,
#    checkpoint_filename=checkpoint_filename,
#    checkpoint_dir=checkpoint_dir,
#    nb_epoch=2,
#    batch_size=32,
#    data_augmentation=False)
##    
#print "Execution time: ", datetime.now() - startTime

#    
##
### plot neural network training log
###plot_save_cnn_results(training_log, accuracy=True, cross_entropy_loss=True, 
###        show_plot=True)
##

# vacancy 10%
#desc_file_list_test = desc_file_list_test_vac10
# vacancy 20%
#desc_file_list_test = desc_file_list_test_vac20
# vacancy 30%
#desc_file_list_test = desc_file_list_test_vac30
# vacancy 40%
desc_file_list_test = desc_file_list_test_vac40
# vacancy 50%
#desc_file_list_test = desc_file_list_test_vac50
##
### disp 0.05A
###desc_file_list_test= desc_file_list_disp05
### disp 0.10A
###desc_file_list_test= desc_file_list_disp10
### disp 0.15A
#desc_file_list_test= desc_file_list_disp15
####
##print "displacements"
##
run_cnn_model(method='xray_cnn', 
    desc_type='xray',
    train=False, 
    split_train_val=True,    
    read_from_file=False,
    path_to_x_train=path_to_x_train,
    path_to_y_train=path_to_y_train,
    path_to_x_val=path_to_x_val,
    path_to_y_val=path_to_y_val,
    path_to_x_test=path_to_x_test,
    path_to_y_test=path_to_y_test,
    target_name='spacegroup_symbol', target_categorical=True,
    desc_file_list_train=desc_file_list_train,
    desc_file_list_test=desc_file_list_test,
    conf_matrix_file=conf_matrix_file,
    input_dims=input_dims,
#    desc_folder=tmp_folder,
    desc_folder=desc_folder,
    tmp_folder=tmp_folder, 
    lookup_file=lookup_file, 
    control_file=control_file,
    results_file=results_file,
#    checkpoint_filename='pristine_all_kernel_3',
    checkpoint_filename=checkpoint_filename,
    checkpoint_dir=checkpoint_dir,
    nb_epoch=1,
    batch_size=32,
    data_augmentation=False)
####
###
#print "Execution time: ", datetime.now() - startTime
##
##
#desc_file_list_test= desc_file_list_test_vac50
####
##print "vacancies 50%"
##
#run_cnn_model(method='xray_cnn', 
#    desc_type='xray',
#    train=False, 
#    split_train_val=True,    
#    read_from_file=False,
#    path_to_x_train=path_to_x_train,
#    path_to_y_train=path_to_y_train,
#    path_to_x_val=path_to_x_val,
#    path_to_y_val=path_to_y_val,
#    path_to_x_test=path_to_x_test,
#    path_to_y_test=path_to_y_test,
#    target_name='spacegroup_symbol', target_categorical=True,
#    desc_file_list_train=desc_file_list_train,
#    desc_file_list_test=desc_file_list_test,
#    conf_matrix_file=conf_matrix_file,
#    input_dims=input_dims,
##    desc_folder=tmp_folder,
#    desc_folder=desc_folder,
#    tmp_folder=tmp_folder, 
#    lookup_file=lookup_file, 
#    control_file=control_file,
#    results_file=results_file,
##    checkpoint_filename='pristine_all_kernel_3',
#    checkpoint_filename=checkpoint_filename,
#    checkpoint_dir=checkpoint_dir,
#    nb_epoch=1,
#    batch_size=32,
#    data_augmentation=False)
####
###
#print "Execution time: ", datetime.now() - startTime

##
#sys.exit(1)

#
##images = images[:1]
#model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename +'.h5')))
#model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename + '.json')))
##
##
#images = extract_images(filename=desc_file, 
#    filetype='descriptor_files',
#    input_dims=input_dims, desc_folder=tmp_folder, tmp_folder=tmp_folder)
#images = np.reshape(images, (-1, input_dims[0], input_dims[1]))
##
##
###    
#plot_att_response_maps(images, model_arch_file, model_weights_file, figure_dir, 
#    nb_conv_layers=6,
#    nb_top_feat_maps=4, 
#    layer_nb='all',
##    layer_nb=[0, 5],
#    plot_all_filters=True,
#    plot_filter_sum=True,
#    plot_summary=True)
#
#
sys.exit(1)

#==============================================================================
# Structural transitions
#==============================================================================

desc_file_bcc_to_sc = [
    'descriptor_all_classes_4_samples.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.0_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.1_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.2_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.3_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.4_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.5_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.6_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.7_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.8_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.9_type_1.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_1.0_type_1.tar.gz']

run_cnn_model(method='xray_cnn', 
    desc_type='xray',
    train=False, 
    split_train_val=True,    
    read_from_file=False,
    path_to_x_train=path_to_x_train,
    path_to_y_train=path_to_y_train,
    path_to_x_val=path_to_x_val,
    path_to_y_val=path_to_y_val,
    path_to_x_test=path_to_x_test,
    path_to_y_test=path_to_y_test,
    target_name='spacegroup_symbol', target_categorical=True,
    desc_file_list_train=None,
    desc_file_list_test=desc_file_bcc_to_sc,
    input_dims=input_dims,
    desc_folder=example_data_bcc_to_sc_folder,
    tmp_folder=tmp_folder, 
    lookup_file=lookup_file, 
    control_file=control_file,
    results_file=results_file_bcc_to_sc,
    checkpoint_filename=checkpoint_filename,
    checkpoint_dir=checkpoint_dir,
    nb_epoch=1,
    batch_size=32,
    data_augmentation=False)
    
df_results = aggregate_struct_trans_data(results_file_bcc_to_sc, nb_samples=125, 
    nb_order_param_steps=11, max_order_param=1.0, 
    prob_idxs=[3, 2])
    
make_crossover_plot(df_results, results_file_bcc_to_sc, 
    prob_idxs=[3, 2], 
    labels = ["$p_{diamond}$", "$p_{fcc}$", "$p_{bcc}$", "$p_{sc}$"],
    nb_order_param_steps=11,     
    filename_suffix=".png", 
    title="Body-centered-cubic (bcc) to simple cubic (sc) crossover", 
    x_label="Central atoms removed [%]", show_plot=False)

make_crossover_plot(df_results, results_file_bcc_to_sc, 
    prob_idxs=[3, 2], 
    labels = ["$p_{diamond}$", "$p_{fcc}$", "$p_{bcc}$", "$p_{sc}$"],
    nb_order_param_steps=11,     
    filename_suffix=".svg", 
    title="Body-centered-cubic (bcc) to simple cubic (sc) crossover", 
    x_label="Central atoms removed [%]", show_plot=False)
    
sys.exit(1)
    
#
#desc_file_diam_to_fcc = [
#    'descriptor_all_classes_4_samples.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.0.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.2.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.3.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.4.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.5.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.6.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.7.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.8.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_0.9.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_1.0.tar.gz']
#
#run_cnn_model(method='xray_cnn', 
#    desc_type='xray',
#    train=False, 
#    split_train_val=True,    
#    read_from_file=False,
#    path_to_x_train=path_to_x_train,
#    path_to_y_train=path_to_y_train,
#    path_to_x_val=path_to_x_val,
#    path_to_y_val=path_to_y_val,
#    path_to_x_test=path_to_x_test,
#    path_to_y_test=path_to_y_test,
#    target_name='spacegroup_symbol', target_categorical=True,
#    desc_file_list_train=None,
#    desc_file_list_test=desc_file_diam_to_fcc,
#    input_dims=input_dims,
#    desc_folder=example_data_diam_to_fcc_folder,
#    tmp_folder=tmp_folder, 
#    lookup_file=lookup_file, 
#    control_file=control_file,
#    results_file=results_file_diam_to_fcc,
#    checkpoint_filename=checkpoint_filename,
#    checkpoint_dir=checkpoint_dir,
#    nb_epoch=1,
#    batch_size=32,
#    data_augmentation=False)
#    
#
#df_results = aggregate_struct_trans_data(results_file_diam_to_fcc, nb_samples=48, 
#    nb_order_param_steps=11, max_order_param=1.0, 
#    prob_idxs=[1, 0])
#
#make_crossover_plot(df_results, results_file_diam_to_fcc, 
#    prob_idxs=[1, 0], 
#    labels = ["$p_{diamond}$", "$p_{fcc}$", "$p_{bcc}$", "$p_{sc}$"],
#    nb_order_param_steps=11,     
#    filename_suffix=".png", 
#    title="Diamond to face-centered-cubic (fcc) crossover", 
#    x_label="Central atoms removed [%]", show_plot=True)


desc_file_bcc_to_amorphous = [
    'descriptor_all_classes_4_samples.tar.gz',
    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',

    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz',
    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz',
    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz']    


run_cnn_model(method='xray_cnn', 
    desc_type='xray',
    train=False, 
    split_train_val=True,    
    read_from_file=False,
    path_to_x_train=path_to_x_train,
    path_to_y_train=path_to_y_train,
    path_to_x_val=path_to_x_val,
    path_to_y_val=path_to_y_val,
    path_to_x_test=path_to_x_test,
    path_to_y_test=path_to_y_test,
    target_name='spacegroup_symbol', target_categorical=True,
    desc_file_list_train=None,
    desc_file_list_test=desc_file_bcc_to_amorphous,
    input_dims=input_dims,
    desc_folder=example_data_bcc_to_amorphous_folder,
    tmp_folder=tmp_folder, 
    lookup_file=lookup_file, 
    control_file=control_file,
    results_file=results_file_bcc_to_amorphous,
    checkpoint_filename=checkpoint_filename,
    checkpoint_dir=checkpoint_dir,
    nb_epoch=1,
    batch_size=32,
    data_augmentation=False)




df_results = aggregate_struct_trans_data(results_file_bcc_to_amorphous, 
    nb_samples=421,
    nb_order_param_steps=9, max_order_param=0.4, 
    prob_idxs=[0, 1, 2, 3])


make_crossover_plot(df_results, results_file_bcc_to_amorphous, 
    prob_idxs=[0, 1, 2, 3], 
    labels = ["$p_{diamond}$", "$p_{fcc}$", "$p_{bcc}$", "$p_{sc}$"],
    nb_order_param_steps=9,     
    filename_suffix=".png", 
    title="From body-centered-cubic (bcc) to amorphous", 
    x_label="Lindemann parameter", show_plot=False)
    

make_crossover_plot(df_results, results_file_bcc_to_amorphous, 
    prob_idxs=[0, 1, 2, 3], 
    labels = ["$p_{diamond}$", "$p_{fcc}$", "$p_{bcc}$", "$p_{sc}$"],
    nb_order_param_steps=9,     
    filename_suffix=".svg", 
    title="From body-centered-cubic (bcc) to amorphous", 
    x_label="Lindemann parameter", show_plot=False)   
    
    
#==============================================================================
# other useful scripts
#==============================================================================

#desc_file_list = []
#for root, dirs, files in os.walk(desc_folder, topdown=True):
#    for file_ in files:
#        if file_.endswith(".tar.gz"):
#            desc_file_list.append(os.path.join(root, file_))
#                            
#print desc_file_list

#for idx, desc_file in enumerate(desc_file_list[173:]):
#    print idx, len(desc_file_list)
#    filename = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file))) 
#    try:
#        write_summary_file(filename, tmp_folder)
#    except Exception:
#        print filename 


#tar_gz_list = []
#
#for root, dirs, files in os.walk(desc_folder, topdown=True):
#    for file_ in files:
#        if file_.endswith(".tar.gz"):
#            tar_gz_list.append(os.path.join(root, file_))





