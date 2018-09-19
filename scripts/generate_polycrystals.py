#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/01/18"


import numpy as np
import os.path
import sys

apt_dir = os.path.normpath("/home/ziletti/nomad/nomad-lab-base/apt")

sys.path.insert(0, apt_dir)

from ai4materials.utils.utils_crystals import get_boxes_from_xyz
from grain_boundaries.generate_polycrystal_and_boxes import generate_polycrystal


main_folder = '/home/ziletti/Documents/calc_xray/rot_inv_3d/polycrystals'

# directories
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp'))) 
checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder'))) 
figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps'))) 

# files
conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png'))) 

data_folder = '/u/ziang/calc_nomad_sim/prototypes_aflow/'


if __name__== "__main__":
    
    ########################
    # Generate polycrystal
    ########################

    output_filename = 'example'

    output_filename_path = os.path.abspath(os.path.normpath(os.path.join(main_folder, output_filename)))

    box_size = [80.0, 80.0, 15.0]
    # four grains
    grain_specifications = [[[40.0, 40.0, 10.56], ['Al', 'fcc', 4.046]], [[0.45, 2.15, 10.56], ['Co', 'hcp', 2.507, 4.067]], [[10.3, 18.1, 10.56],  ['C', 'diamond', 3.571]], [[18.2, 5.21, 10.56], ['Fe', 'bcc', 2.856]]]

    # this requires atomsk installed (http://atomsk.univ-lille1.fr/)
    generate_polycrystal(output_filename, box_size, grain_specifications)
    
    
    ########################
    # Get boxes
    ########################
    
    sliding_volume = [10.0, 10.0, 15.0]
    stride_size = [8.0, 8.0, 15.0]
    xyz_boxes = get_boxes_from_xyz(output_filename+'.xyz', sliding_volume, stride_size)

    boxes = np.asarray(xyz_boxes)
    
    print("Boxes.shape: {}".format(boxes.shape))
    
    #Visualization
    #os.system('ovito '+output_filename+'.xyz')

