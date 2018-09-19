#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "15/11/17"


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
    
from nomad_sim.wrappers import get_json_list, calc_descriptor 
from nomad_sim.utils_crystals import interpolate_parameters
import numpy as np
import math
from nomad_sim.utils_crystals import create_supercell_by_nb_atoms
from nomad_sim.utils_parsing import read_write_json_files_nomad
from nomad_sim.utils_data_retrieval import generate_facets_input
import scipy.optimize
from functools import partial

from datetime import datetime
from nomad_sim.descriptors import Diffraction2D

startTime = datetime.now()
now = datetime.now()

# add / at the end
main_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/'

# data folder needs to be a list
#data_folder = [data_folder[0]]
 
#directories
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp'))) 
checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder'))) 
figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps'))) 

#files
conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png'))) 
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results.csv'))) 
desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder,'desc_info.json.info'))) 
lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'lookup.dat'))) 
control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'control.json')))
results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results.csv'))) 
filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder,'filtered_file.json'))) 
training_log_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir,'training_'+str(now.isoformat())+'.log'))) 
results_file_bcc_to_sc = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_sc.csv'))) 
results_file_diam_to_fcc = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_diam_to_fcc.csv'))) 
results_file_bcc_to_amorphous = os.path.abspath(os.path.normpath(os.path.join(main_folder,'results_crossover_bcc_to_amorphous.csv'))) 

data_folder=[
    '/u/ziang/parsed/production/VaspRunParser1.2.0-3-g4facbeb/Rnh_4DFTJQgTSOib4e4d-5GByiTVB',
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

# =============================================================================
#  Define descriptor
# =============================================================================
desc_names = [item.split("/")[-1] for item in data_folder]

user_param_source = {
    'wavelength': 5.0E-12,     # best overall till now
    'pulse_energy': 1E-6,  #not important
    'focus_diameter': 1E-6 #not important
    }

user_param_detector = {  
    'distance': 0.1,       #on this scale it is not very important
    'pixel_size': 4E-4,    
    'nx': 64,
    'ny': 64
}

input_dims = (64, 64)

kwargs = {'ndim': 2,
          'user_param_source': user_param_source,
          'user_param_detector': user_param_detector,
#          'atoms_scaling': 'min_distance_nn'}
          'atoms_scaling': 'avg_distance_nn',
          'use_mask': True}
cell_type = 'standard'

descriptor = Diffraction2D(use_autoencoder=False, **kwargs)

# =============================================================================
# Define operations on structures
# =============================================================================
operations_on_structure_list = [
    (create_supercell_by_nb_atoms, {'min_nb_atoms': 32, 'target_nb_atoms': 256})

]

operation_names = [
    "_supercell_by_nb_atoms_min32_max256_pristine"
    ]

#a = 5.0
#c_a = math.sqrt(2)

# =============================================================================
# BCC -> RH -> SC -> FCC
# =============================================================================
## parameters taken from http://www.aflowlib.org/CrystalDatabase/A_oC4_63_c.html
## vectors: a, b/a, c/a, y
## as required by the aflow online generator
#hcp_params = np.array([a, math.sqrt(3), c_a, 1.0/6.0])
#bcc_params = np.array([a, math.sqrt(2), math.sqrt(2), 1.0/4.0])
#fcc_params = np.array([a, 1.0, 1.0, 1.0/4.0])
#sc_params = np.array([a, 1.0, 2.0, 0.0])
a= 5.0
c_a=1.5

# http://www.aflowlib.org/CrystalDatabase/A_hR1_166_a.beta-Po.html
# beta-Po: c_a=0.968139947937

alpha = np.linspace(80, 100, 1000, retstep=False)
print alpha


def f_all(c_a, alpha):    
    return math.acos((2.0*c_a**2 - 3.0)/(2.0*(c_a**2 + 3))) - math.radians(alpha)

for alpha_v in alpha:
    c_a0 = 1.0
    f = partial(f_all, alpha=alpha_v)
        
    sol = scipy.optimize.root(f, c_a0, method='lm')

    print "alpha:", alpha_v, "c_a:", sol.x, "success:", sol.success, np.sum(f(sol.x))


np.set_printoptions(precision=8)
    
sys.exit(1)


print "Bcc to rh to sc to rh to fcc transition"
const = math.sqrt(3.0/8.0)
#const = 1.0
# rh_params_1 = np.array([0.50*math.sqrt(3.0/8.0)])/const
rh_params_1 = np.array([1.*math.sqrt(3.0/8.0)])/const
bcc_params = np.array([math.sqrt(3.0/8.0)])/const
sc_params = np.array([2.0*math.sqrt(3.0/8.0)])/const
fcc_params = np.array([4.0*math.sqrt(3.0/8.0)])/const
rh_params_2 = np.array([5.0*math.sqrt(3.0/8.0)])/const


seg_0 = interpolate_parameters(rh_params_1, bcc_params, nb_steps=0,
    include_final=False)
seg_1 = interpolate_parameters(bcc_params, sc_params, nb_steps=40,
    include_final=False)
seg_2 = interpolate_parameters(sc_params, fcc_params, nb_steps=80,
    include_final=False)
seg_3 = interpolate_parameters(fcc_params, rh_params_2, nb_steps=40,
    include_final=True)
full_transition = seg_0 + seg_1 + seg_2 + seg_3

for idx, param in enumerate(full_transition):
    print idx, param, param*const

input_generate_param = [item*const for item in full_transition]

print np.concatenate(input_generate_param).ravel().tolist()

sys.exit(1)

print "Bct 141 to diamond transition"
const = math.sqrt(2.0)
bct_param_1 = np.array([0.50*math.sqrt(2.0)])/const
diam_params = np.array([math.sqrt(2.0)])/const
bct_params_2 = np.array([1.5*math.sqrt(2.0)])/const
np.set_printoptions(precision=8)

seg_0 = interpolate_parameters(bct_param_1, diam_params, nb_steps=5, 
    include_final=False)

seg_1 = interpolate_parameters(diam_params, bct_params_2, nb_steps=5, 
    include_final=True)

full_transition = seg_0 + seg_1 
for idx, param in enumerate(full_transition):
    print idx, param, param*const


# =============================================================================
# BCT -> BCC -> FCC
# =============================================================================

data_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/rh_bcc_sc_fcc_rh'
output_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/rh_bcc_sc_fcc_rh'

#data_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_bcc_fcc_bct/'
#output_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_bcc_fcc_bct/'

# data_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_diam/'
# output_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_diam/'


# read_write_json_files_nomad(data_folder, output_folder, NB_MAX_FOLDERS=1000, NB_MAX_FILES=100)

# json_list = get_json_list(method='folder', data_folder=output_folder)

# the list needs to be ordered because we want to plot a transition

#json_list = [
#'0_nomad.json',
#'1_nomad.json',
#'2_nomad.json',
#'3_nomad.json',
#'4_nomad.json',
#'5_nomad.json',
#'6_nomad.json',
#'7_nomad.json',
#'8_nomad.json',
#'9_nomad.json',
#'10_nomad.json',
#'11_nomad.json',
#'12_nomad.json',
#'13_nomad.json',
#'14_nomad.json',
#'15_nomad.json',
#'16_nomad.json',
#'17_nomad.json',
#'18_nomad.json'
#]

json_list = []

suffix_file = "_fig4_paper_nomad.json"
for i in range(0, 161):
    json_list.append(str(i)+suffix_file)

print json_list


json_list_full_path = [os.path.abspath(os.path.normpath(os.path.join(output_folder, item))) for item in json_list]

# =============================================================================
# Rotation matrix for each channel 
# =============================================================================
desc_angles = {"r": [-45.0, 45.0], "g": [-45.0, 45.0], "b": [-45.0, 45.0]}

rot_matrices = {}
rot_matrices_x = []
for angle in desc_angles["r"]:
    rot_matrices_x.append(np.asarray([[1, 0, 0], [0, math.cos(np.radians(angle)), -math.sin(np.radians(angle))], [0, math.sin(np.radians(angle)), math.cos(np.radians(angle))]]))
rot_matrices["r"] = rot_matrices_x

rot_matrices_y = []
for angle in desc_angles["g"]:
    rot_matrices_y.append(np.asarray([[math.cos(np.radians(angle)), 0, math.sin(np.radians(angle))], [0, 1, 0], [-math.sin(np.radians(angle)), 0, math.cos(np.radians(angle))]]))
rot_matrices["g"] = rot_matrices_y

rot_matrices_z = []
for angle in desc_angles["b"]:
    rot_matrices_z.append(np.asarray([[math.cos(np.radians(angle)), -math.sin(np.radians(angle)), 0], [math.sin(np.radians(angle)), math.cos(np.radians(angle)), 0], [0, 0, 1]]))
rot_matrices["b"] = rot_matrices_z

# =============================================================================
# Descriptor calculation - serial execution
# =============================================================================

#json_file_name = [item.rsplit('.',1)[0] for item in json_list]
#desc_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_bcc_fcc_bct'
#desc_folder = '/home/ziletti/Documents/calc_xray/2d_nature_comm/bct_diam'

desc_file_list = []
for idx, json_file in enumerate(json_list_full_path):
    calc_descriptor(desc_type='xray', file_format='NOMAD',
        json_list=json_file, tmp_folder=tmp_folder,
        desc_folder=desc_folder,
        #desc_folder=tmp_folder,
       desc_file='rh_bcc_sc_fcc_'+json_list[idx] + '.tar.gz',
#        desc_file='bct_bcc_fcc_'+json_list[idx] + '.tar.gz',
#         desc_file='bct_diam_'+json_list[idx] + '.tar.gz',
        desc_info_file=desc_info_file,
        rot_matrices=rot_matrices,
        # stupid but works
        target_list=np.zeros(len(json_list)), 
        cell_type=cell_type,
        operations_on_structure=operations_on_structure_list[0],
        **kwargs)

    desc_file_list.append('rh_bcc_sc_fcc_'+json_list[idx] + '.tar.gz')

print desc_file_list

for desc_file in desc_file_list:
    desc_file_path = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))

df_filepath = generate_facets_input(desc_folder=desc_folder, main_folder=main_folder, 
    input_dims=input_dims, desc_file_list=desc_file_list, tmp_folder=tmp_folder)


    
sys.exit(1)