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

if not common_dir in sys.path:
    sys.path.insert(0, common_dir) 
    sys.path.insert(0, nomad_sim_dir) 
    sys.path.insert(0, atomic_data_dir) 
    
# hack to get it running locally - TO BE REMOVED
#sys.path.append(os.path.join(os.path.dirname(__file__), '../', 'python-modules'))

#print sys.path


import tarfile
import json
import numpy as np
from datetime import datetime

from nomad_sim.wrappers import get_json_list, calc_descriptor 
from nomad_sim.wrappers import calc_embedding, plot
from nomad_sim.utils_crystals import create_supercell
from nomad_sim.utils_crystals import spacegroup_a_to_spacegroup_b
from nomad_sim.utils_crystals import create_supercell_by_radius
from nomad_sim.utils_crystals import create_supercell_by_nb_atoms
from nomad_sim.utils_crystals import create_vacancies
from nomad_sim.utils_crystals import random_displace_atoms
from nomad_sim.utils_crystals import substitute_atoms
from nomad_sim.utils_crystals import filter_json_list
from nomad_sim.utils_plotting import plot_bcc_to_scc, plot_diamond_to_fcc, plot_bcc_disorder

from nomad_sim.descriptors import XrayDiffraction
from nomad_sim.model_cnn import run_cnn_model

startTime = datetime.now()


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

# add / at the end
main_folder = '/home/ziletti/Documents/calc_xray/faces_crystals_tutorial/'


# data folder needs to be a list
#data_folder = [data_folder[0]]
data_folder = data_folder[0:2]
 
tmp_folder = main_folder
desc_folder = main_folder +'/desc_folder/'
#desc_folder = main_folder +'/desc_bck/'
checkpoint_dir = main_folder + '/saved_models/'

desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder,'desc_info.json.info'))) 
lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'lookup.dat'))) 
control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'control.json'))) 
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results.csv'))) 
      



user_param_source = {
    'wavelength': 5.0E-12,     # best overall till now
    'pulse_energy': 1E-6,  #not important
    'focus_diameter': 1E-6 #not important
    }

user_param_detector = {  
    'distance': 0.1,       #on this scale it is not very important
    # for 64x64
    'pixel_size': 4E-4,    # the right one
    'nx': 64,
    'ny': 64

    # for 640x640
#    'pixel_size': 4E-5,    
#    'nx': 640,
#    'ny': 640

}

#for 2d
input_dims = (64, 64)

#angles_d = [45.0]

angles_d = [45.0]

#print 'angle', angles_d
kwargs = {'ndim': 2,
          'desc_space': 'k-space', 
#          'desc_space': 'r-space', 
          'user_param_source': user_param_source,
          'user_param_detector': user_param_detector,
          'angles_d': angles_d,
          'rotation': True,
#          'atoms_scaling': 'min_distance_nn'}
          'atoms_scaling': 'avg_distance_nn'}

cell_type = 'standard'
#cell_type = 'primitive

operations_on_structure = [
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 64}),
#    (random_displace_atoms, {'displacement': 0.15, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),
#    (random_displace_atoms, {'displacement': 0.20, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 64}),

#    (create_supercell, {'replicas': [1, 1, 1]}),
#    (create_supercell, {'replicas': [1, 1, 1]}),
#    (create_supercell, {'replicas': [1, 1, 1]}),
#    (create_supercell, {'replicas': [1, 1, 1]}),
#    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 128}),

#    (spacegroup_a_to_spacegroup_b, {'spgroup_a': 'Im-3m', 'spgroup_b': 'Pm-3m', 'target_b_contribution': 0.0, 'create_replicas_by': 'nb_atoms', 'min_nb_atoms': 32, 'target_nb_atoms': 128}),
#    (spacegroup_a_to_spacegroup_b, {'spgroup_a': 'Im-3m', 'spgroup_b': 'Pm-3m', 'target_b_contribution': 0.8, 'create_replicas_by': 'nb_atoms', 'min_nb_atoms': 32, 'target_nb_atoms': 128}),
#    (spacegroup_a_to_spacegroup_b, {'spgroup_a': 'Im-3m', 'spgroup_b': 'Pm-3m', 'target_b_contribution': 1.0, 'create_replicas_by': 'nb_atoms', 'min_nb_atoms': 32, 'target_nb_atoms': 128}),

#    (spacegroup_a_to_spacegroup_b, {'spgroup_a': 'Fd-3m', 'spgroup_b': 'Fm-3m', 'target_b_contribution': 1.0, 'create_replicas_by': 'nb_atoms', 'min_nb_atoms': 32, 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.05, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.10, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.15, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.20, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.25, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.30, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.35, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.40, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.45, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.50, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.55, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.60, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.65, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.70, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.75, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.80, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.85, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.90, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.95, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 1.00, 'noise_distribution': 'gaussian_scaled', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

#    (spacegroup_a_to_spacegroup_b, {'spgroup_a': 'Im-3m', 'spgroup_b': 'Pm-3m', 'target_b_contribution': 1.0, 'create_replicas_by': 'nb_atoms', 'min_nb_atoms': 32, 'target_nb_atoms': 128}),

#    (create_supercell, {'replicas': [3, 3, 3]}),

    (create_supercell, {'replicas': [1, 1, 1]}),
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 128}),

#    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 1024}),
#    (random_displace_atoms, {'displacement': 0.01, 'noise_distribution': 'gaussian', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

#    (random_displace_atoms, {'displacement': 0.40, 'noise_distribution': 'gaussian', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

    (create_vacancies, {'target_vacancy_ratio': 0.50, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
#    (random_displace_atoms, {'displacement': 0.15, 'noise_distribution': 'gaussian', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 128}),

    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 64}),
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 64}),
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 64}),
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 64}),

#    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 128}),
#    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 256}),

#    (create_vacancies, {'target_vacancy_ratio': 0.10, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),
    (create_vacancies, {'target_vacancy_ratio': 0.50, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

#    (substitute_atoms, {'target_sub_ratio': 0.75, 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128, 'max_n_sub_species': 2}),

    (random_displace_atoms, {'displacement': 0.15, 'noise_distribution': 'gaussian', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

#    (random_displace_atoms, {'displacement': 0.50, 'noise_distribution': 'uniform', 'create_replicas_by': 'nb_atoms', 'target_nb_atoms': 128}),

]

del operations_on_structure[1:]

descriptor = XrayDiffraction(use_autoencoder=False, **kwargs)
desc_names = [item.split("/")[-1] for item in data_folder]

##

start_desc=0

# filter json files
for i in range(start_desc, len(desc_names)):
    json_list = []

    json_list.append(get_json_list(method='folder', drop_duplicates=False, 
        data_folder=data_folder[i], tmp_folder=tmp_folder))
        
    json_list = [item for sublist in json_list for item in sublist]
    #[:100] will give already the four classes with Rnh
#    json_list = json_list[:10]
    # filtering the json_list
    json_list = filter_json_list(file_format='NOMAD',
    json_list=json_list, tmp_folder=tmp_folder,
    desc_folder=desc_folder,
    write_to_file=True,
    filtered_file=desc_names[i]+'_Pm-3m'+'.json.filter',
    #desc_folder=tmp_folder,
    filter_by=['lattice_type', 'spacegroup_symbol'],
    cell_type=cell_type,
#    accepted_labels=[['cubic'], ['Fd-3m']],
#    accepted_labels=[['cubic'], ['Im-3m']],
#    accepted_labels=[['cubic'], ['Fm-3m']],
    accepted_labels=[['cubic'], ['Pm-3m']],
#    accepted_labels=[['cubic'], ['Fd-3m', 'Fm-3m', 'Im-3m', 'Pm-3m']],
    operations_on_structure=operations_on_structure,
    **kwargs) 



    
#with open(filtered_file) as json_file:
#    try:
#        data = json.load(json_file)
#        print data["data"][0]["filtered_json_list"]
#    finally:
#        json_file.close()
    
    
#    print 'calculating descriptor', i
#    calc_descriptor(desc_type='xray', file_format='NOMAD',
#        json_list=json_list, tmp_folder=tmp_folder,
#        desc_folder=desc_folder,
#        #desc_folder=tmp_folder,
##        desc_file='descriptor.tar.gz',
##        desc_file=desc_names[i]+'_bcc_only_supercell_by_nb_atoms_min32_max128_diam_to_fcc_1.0'+'.tar.gz',
##        desc_file=desc_names[i]+'_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40'+'.tar.gz',
#        desc_file='try_geo'+'.tar.gz',
#        desc_info_file=desc_info_file,
#        grayscale=True,
#        # stupid but works
#        target_list=np.zeros(len(json_list)), 
#        cell_type=cell_type,
#        operations_on_structure=operations_on_structure,
#        **kwargs) 

sys.exit(1)
#        
#    
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
## works only if desc_file_list_test is one element list
desc_file_list_test = [
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

#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.00.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.05.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.10.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.15.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.20.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.25.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.30.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.35.tar.gz',
#
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz',
#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz',
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.40.tar.gz']    


#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.0.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.7.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_1.0.tar.gz']

#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.45.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_gauss_scaled_disp_0.50.tar.gz']

    
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.0_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.1_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.2_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.3_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.4_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.5_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.6_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.7_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.8_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_0.9_type_1.tar.gz',
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_bcc_only_supercell_by_nb_atoms_min32_max128_bcc_to_sc_1.0_type_1.tar.gz']
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

#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128.tar.gz']
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_50.tar.gz']
#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_vacancy_80.tar.gz']

#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_1.0A.tar.gz']

#    'Rnh_4DFTJQgTSOib4e4d-5GByiTVB_supercell_by_nb_atoms_min32_max128_gauss_disp_0.15A.tar.gz']

#    'R10ncY1AZG6X9y-Nj8F0_DiN8NeLD_supercell_by_nb_atoms_min32_max128.tar.gz']
#    'RsLoZhSAdK0BopfI2T4B5pLfMyjVN_supercell_by_nb_atoms_min32_max128.tar.gz',
#    'RMGpPc3B_HiR0D-oLE4ND66HmYdH-_supercell_by_nb_atoms_min32_max128.tar.gz']
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
#    'Ra8nAuJOgxGwSytw1scU5BTeB3ozo_supercell_by_nb_atoms_min32_max128.tar.gz']
    

#desc_file_list_train = desc_file_list_train[:3]
#desc_file_list_test = desc_file_list_test[:3]

tar_gz_list = []

for root, dirs, files in os.walk(desc_folder, topdown=True):
    for file_ in files:
        if file_.endswith(".tar.gz"):
            tar_gz_list.append(os.path.join(root, file_))


path_to_x_train = checkpoint_dir + 'data_set_xtrain.pkl'
path_to_y_train = checkpoint_dir + 'data_set_ytrain.pkl'
path_to_x_val = checkpoint_dir + 'data_set_xval.pkl'
path_to_y_val = checkpoint_dir + 'data_set_yval.pkl'
path_to_x_test = checkpoint_dir + 'data_set_xtest.pkl'
path_to_y_test = checkpoint_dir + 'data_set_ytest.pkl'


#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv'))) 
#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc_type1.csv'))) 
#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_bcc_disorder.csv'))) 
#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_bcc_disorder_all.csv'))) 
#
##
target_pred_class, target_pred_probs, num_labels, class_labels = run_cnn_model(method='xray_cnn', 
    desc_type='xray',
    train=False, 
    # set always True for now
    split_train_val=True,    
    read_from_file=False,
    path_to_x_train=path_to_x_train,
    path_to_y_train=path_to_y_train,
    path_to_x_val=path_to_x_val,
    path_to_y_val=path_to_y_val,
    path_to_x_test=path_to_x_test,
    path_to_y_test=path_to_y_test,
    #target_name='target', target_categorical=True,
#    target_name='Bravais_lattice_lt', target_categorical=True,
#    target_name='lattice_centering', target_categorical=True,
    target_name='spacegroup_symbol', target_categorical=True,
    desc_file_list_train=desc_file_list_train,
    desc_file_list_test=desc_file_list_test,
#    desc_file_list_test='descriptor.tar.gz',
    input_dims=input_dims,
    desc_folder=desc_folder,
    #desc_folder=tmp_folder,
    tmp_folder=tmp_folder, lookup_file=lookup_file, 
    control_file=control_file,
    results_file=results_file,
    #checkpoint_dir=desc_folder,
    checkpoint_filename='pristine_all_kernel_7_final',
    checkpoint_dir=checkpoint_dir,
    nb_epoch=2,
    batch_size=32,
    data_augmentation=True)


# read panda dataframe and plot results
#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv'))) 
#plot_bcc_to_scc(results_file, show_plot=True)

#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_diamond_fcc.csv'))) 
#plot_diamond_to_fcc(results_file, show_plot=True)

#results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_bcc_disorder_all.csv'))) 
#plot_bcc_disorder(results_file, show_plot=True)

print "Execution time: ", datetime.now() - startTime

#sys.exit(1)


target_pred_list = target_pred_probs
target_list = num_labels

embed_params={'learning_rate': 500}

model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7_final.h5')))
model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7_final.json')))

#model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7.h5')))
#model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir + 'pristine_all_kernel_7.json')))

# embedding is not compatible with desc_file being a list
#calc_embedding(embed_method='pca', desc_type='xray', 
calc_embedding(embed_method='tsne_pca', desc_type='xray', 
#calc_embedding(embed_method='tsne', desc_type='xray', 
#calc_embedding(embed_method='spectral_embedding', desc_type='xray', 
#calc_embedding(embed_method='mds', desc_type='xray', 
#    lookup_file=lookup_file, desc_file=desc_file_list_test[0],
    target_categorical=True,
    input_dims=input_dims,
    lookup_file=lookup_file, desc_file=desc_file_list_test,
    #desc_folder=tmp_folder, 
    desc_folder=desc_folder,
    standardize='True',
    target_name='target', embed_params=embed_params, 
    use_xray_img=True,
    model_arch_file=model_arch_file, 
    model_weights_file=model_weights_file,
    nb_nn_layer=-1,
    path_to_x_test=path_to_x_test
    )
    

json_list, frame_list, x_list, y_list, foo = get_json_list(method='file', data_folder=data_folder,
        #path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
        path_to_file=lookup_file, drop_duplicates=False, displace_duplicates=False, get_unique_list=True)
        

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


#print 'original target list', target_list, len(target_list)

# read from desc_file
#try:
#    with open(desc_info_file) as data_file:
#        data = json.load(data_file)
#
#        for c in data['descriptor_info']:
#            xray_img_list = c["xray_img_list"]
#except:
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


#for idx, item in enumerate(target_class):
#    print idx, target_list[idx], target_pred_class[idx]


plot_misclassified_only = False

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

#    print json_list

# only for visualization
# by default the operations defined before the descriptor calculation are used    
#operations_on_structure = [
##    (create_vacancies, {'vacancy_ratio': 0.1, 'replicas': [3, 3, 3]})]
##    (random_displace_atoms, {'displacement': 0.20, 'replicas': [2, 2, 2]})]
#    (create_supercell, {'replicas': [1, 1, 1]})]

filename = plot(
    name='xray_plot', 
    json_list=json_list, frames='list', frame_list=frame_list, op_list=op_list, 
    descriptor=descriptor,
    operations_on_structure=operations_on_structure,
    xray_img_list=xray_img_list,
    file_format='NOMAD', clustering_x_list=x_list, clustering_y_list=y_list, 
    target_list=target_list,
    is_classification=True,
    target_class_names=classes,
    target_pred_list=target_pred_list, target_unit='',
    clustering_point_size=12, tmp_folder=tmp_folder, control_file=control_file,
    cell_type=cell_type,
    atoms_scaling='avg_distance_nn'
    )





# ------------------------------ OLD STUFF ------------------------------ '''

#
#angles_d = []
#for i in range(0, 360, 15):
#    angles_d.append(i)
#    
#print 'Calculating desc with the following angles:', angles_d    
#
#for angle in angles_d:
#    cell_type='standard'
#
#    print 'Calculating angle ', angle
#    kwargs = {'user_param_source': user_param_source,
#          'user_param_detector': user_param_detector,
#          'angles_d': [angle],
#          'rotation': True}
#          #'cell_type': 'standard'}
#          #'cell_type': 'primitive'}
#
#    calc_descriptor(desc_type='xray', file_format='NOMAD',
#        json_list=json_list_bcc_fcc_hex, tmp_folder=tmp_folder,
#        desc_file='descriptor_period_6_angle'+str(int(angle))+'_s.tar.gz',
#        grayscale=True,
#        target_list=target_list, 
#        cell_type=cell_type,
#        **kwargs)  







