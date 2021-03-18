# coding=utf-8
# Copyright 2021 Andreas Leitherer
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
from __future__ import division
from __future__ import print_function

__author__ = "Andreas Leitherer"
__copyright__ = "Copyright 2021, Andreas Leitherer"
__maintainer__ = "Andreas Leitherer"
__email__ = "leitherer@fhi-berlin.mpg.de"
__date__ = "16/03/21"

import os
from collections import defaultdict
from copy import deepcopy
import numpy as np
from ai4materials.utils.utils_config import get_data_filename
import ase
from keras.models import load_model
from ase.io import read


import matplotlib
import matplotlib.pyplot as plt

from ai4materials.models.strided_pattern_matching import make_strided_pattern_matching_dataset
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger
from ai4materials.models.strided_pattern_matching import get_classification_map
from ai4materials.utils.utils_data_retrieval import clean_folder
from ai4materials.utils.utils_crystals import get_boxes_from_xyz
import os    
from ai4materials.descriptors.quippy_soap_descriptor import quippy_SOAP_descriptor
import time
import shutil
import pandas as pd


import numpy as np
import json
import pickle


def load_all_prototypes(datasets_to_load = ['Elemental_solids', 'Binaries', 'Ternaries', 'Quaternaries', '2D_materials', 'Nanotubes'],
                        min_nb_atoms = 100,
                        periodic_boundary_conditions = [True], adjust_2D_cell=False):
    """
    
    datasets_to_load: list, default set to load all datasets used in Leitherer et. al. 2021
        List of datasets (Elemental solids etc.) which one wants to load    
        
    min_nb_atoms: int, default 100
        Minimum number of atoms used in supercell calculation. If None, no supercells are determined. 
        The results are saved in the .info dict of the ase object. 
    periodic_boundary_conditions: list, boolean
        List of periodic boundary conditions to be used. If pbc=False, then one should choose a min_nb_atoms,
        otherwise the structures will be too small (and correspond to the number of atoms in the unit cell)
    adjust_2D_cell: boolean, optional, default False
        If True, the cell dimension of 2D materials will be exended to make sure that no atoms from the replicas fall into the SOAP radius.
        Most of the time this will not cause trouble and our policy is to use the structures as they come from the databses.
    """
    
    appendix_to_descfiles = '' 
    # only needed if compute for instance over several nodes -> useful in future release
    
    
    prototypes_path = get_data_filename('data/PROTOTYPES')
    all_datasets = os.listdir(prototypes_path)
    
    datasets = [_ for _ in all_datasets if _ in datasets_to_load]
    
    all_prototypes = []
        
    for p_b_c in periodic_boundary_conditions:
            for dataset in datasets:
                #if not dataset=='2D_materials':
                #    continue
                # skip Nanotubes if pbc True
                # NOT DO THE FOLLOWING when only computing pbc=True!
                if p_b_c==True and dataset=='Nanotubes':
                    continue
                # because it will skip nanotubes and then only return 96 instead of 108 classes
                # which we do not want
                #if not dataset=='Elemental_solids':
                #    continue
                # HACK to only calculate El.sol.,2D mats for pbc=True and all materials for pbcTrue
                #if p_b_c==True and (dataset=='2D_materials' or dataset=='Elemental_solids'):
                #    pass
                #elif p_b_c==False:
                #    pass
                #else:
                #    continue
                # HACK to get only 2D mats for pbc True and False
                #if not dataset=='2D_materials':
                #    continue            
    
                prototypes_basedir = os.path.join(prototypes_path,dataset)
                crystal_structures = os.listdir(prototypes_basedir)
                
                # LOAD PROTOTYPES
                # go trhough folder, get all txt files, load all of protos, give them labels according to their filename which will identify them uniquely  
                prototypes = []
                for crystal_structure in crystal_structures:
                    all_files = os.listdir(os.path.join(prototypes_basedir, crystal_structure))
                    all_files = [x for x in all_files if x[-3:]=='.in'] # before: only consider txt files. Now changed that to .in
                    for i,filename in enumerate(all_files):
                        prototype = ase.io.read(os.path.join(prototypes_basedir,crystal_structure,filename), ':', 'aims')[0]
                        prototype.info['idx'] = i
                        prototype.info['label'] = appendix_to_descfiles+'_'+filename[:-3]+'_pbc_'+str(p_b_c) # label for assigning unique name to all files in the tar files
                        prototype.info['crystal_structure'] = crystal_structure # THIS WILL BE USED AS TARGET FOR TRAINING
                        prototype.info['dataset'] = dataset
                        prototype.info['geo_file_location'] = os.path.join(prototypes_basedir,crystal_structure,filename)
                        # Elongate cell such that do not run into trouble for pbc=True
                        if adjust_2D_cell and dataset=='2D_materials':
                            structure = deepcopy(prototype)
                            max_thickness = 15.1
                            max_scale_factor = 5.5
                            max_cutoff = 5.0 * max_scale_factor
                            tolerance = 2.0 * max_scale_factor # for safety
                            # starting from mean of cell, have to add half of the thickness (since later shift the layers such that mean of positions is at mean of cell)
                            # and then add maximum cutoff size (eg 5.0 * maximum scaling factor) plus some tolerance factor for safety
                            max_cell_size = max_cutoff + max_thickness/2. + tolerance # intially multiplied by 2, but have empty space on both sides of cell, so not required!
                            print('Maximal required cell size = {}'.format(max_cell_size))
                            
                            # shift mean of atoms to mean of cell
                            original_mean_structure_positions_z = np.mean(structure.positions[:,-1])
                            mean_new_cell_z = max_cell_size / 2.0 #structure.cell[-1][-1]  / 2.
                            difference_means = abs(original_mean_structure_positions_z - mean_new_cell_z)
                            
                            new_positions = structure.positions + [0.0, 0.0, difference_means]
                            
                            new_structure = deepcopy(structure)
                            new_structure.cell[-1][-1] = max_cell_size 
                            new_structure.set_positions(new_positions)   
                            
                            if adjust_2D_cell:
                                structure = deepcopy(new_structure)
                            prototype = deepcopy(structure)
                        
                        # get supercell size - replicate unit cell until min_nb_atoms is exceeded
                        if dataset=='2D_materials': # for 2D materials only replicate in x and y
                            replica = 0
                            while (prototype*(replica,replica,1)).get_number_of_atoms()<min_nb_atoms:
                                replica+=1
                            supercell_list = []
                            for x in [replica, replica+1, replica+2]:
                                for y in [replica, replica+1, replica+2]:
                                    supercell_list.append((x,y,1))
                            prototype.info['supercells'] = [(x,x,1) for x in [replica, replica+1, replica+2]] # supercell_list
                        elif dataset=='Nanotubes':
                            prototype.info['supercells'] = ['Nanotubes']
                        else:
                            replica = 0
                            while (prototype*(replica,replica,replica)).get_number_of_atoms()<min_nb_atoms:
                                replica+=1
                            supercell_list = []
                            for x in [replica,replica+1]:
                                for y in [replica, replica+1]:
                                    for z in [replica, replica+1]:
                                        supercell_list.append((x,y,z))
                            prototype.info['supercells'] = [(x,x,x) for x in [replica,replica+1, replica+2]] # supercell_list                    
                            #prototype.info['supercells'].append(['cubic_shape_1','cubic_shape_2'])
                            
                        # channged to fit in pbc=True
                        #prototype.set_pbc(False)
                        if dataset=='Nanotubes':
                            prototype.set_pbc(False)
                        elif dataset=='2D_materials':
                            prototype.set_pbc([p_b_c,p_b_c,p_b_c])#False])
                        else:
                            prototype.set_pbc(p_b_c)
                        prototypes.append(prototype)
                        all_prototypes.append(prototype)
    return all_prototypes
    
    
    




def plot_prediction_heatmaps_(prob_prediction_class, title, main_folder, class_name='', prefix='prob', suffix='',
                             cmap='viridis', color_nan='lightgrey', interpolation='none', vmin=None, vmax=None):
    """

    For available interpolation methods see:
    https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html

    """

    if len(prob_prediction_class.shape) == 2:
        #logger.info("Creating two-dimensional plot.")
        fig, ax = plt.subplots()

        cmap = matplotlib.cm.get_cmap(name=cmap)
        # set the color for NaN values
        cmap.set_bad(color=color_nan)

        cax = ax.imshow(prob_prediction_class, interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap,
                        origin='lower')

        if class_name != '':
            ax.set_title('{} for class {}'.format(title, class_name))
        else:
            ax.set_title('{}'.format(title))

        # plt.xlabel(u'x  [\u212B]')
        # plt.ylabel(u'y [\u212B]')
        plt.xlabel(u'x stride #')
        plt.ylabel(u'y stride #')

        # add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax)
    else:

        # from mayavi import mlab

        azimuth = 0.0
        elevation = 0.0
        # mlab.options.offscreen = False
        # mlab.clf()

        # obj = mlab.contour3d(prob_prediction_class, contours=100, vmin=0.2, vmax=1., opacity=0.)

        filename_npy = os.path.join(main_folder,
                                    '{0}_{1}_class{2}.npy'.format(str(title), str(prefix), str(class_name)))

        np.save(filename_npy, prob_prediction_class)
        logger.info("Voxel array containing the classification map saved at {}.".format(filename_npy))
        # obj.scene.disable_render = True  # obj.scene.anti_aliasing_frames = 0  # mlab.view(azimuth=azimuth, elevation=elevation)  # mlab.colorbar(title='Field intensity', orientation='vertical')  # mlab.show()  #  # mlab.savefig(filename=os.path.join(main_folder, '{0}_class{1}.png'.format(str(prefix),  #                                                                           str(class_name))))  # mlab.close(all=True)

        # logger.info("Creating three-dimensional plot.")  # x = np.arange(prob_prediction_class.shape[0])[:, None, None]  # y = np.arange(prob_prediction_class.shape[1])[None, :, None]  # z = np.arange(prob_prediction_class.shape[2])[None, None, :]  # x, y, z = np.broadcast_arrays(x, y, z)  #  # from mpl_toolkits.mplot3d import Axes3D  # # ax = fig.add_subplot(111, projection='3d')  #  # colmap = cm.ScalarMappable(cmap=plt.cm.Blues)  # colmap.set_array(prob_prediction_class.ravel())  #  # fig = plt.figure(figsize=(8, 6))  # ax = fig.gca(projection='3d')  # ax.scatter(x, y, z, marker='s', s=140, c=prob_prediction_class.ravel(), cmap=plt.cm.Blues, vmin=0, vmax=1,  #            alpha=0.7)  # alpha is transparancey value, 0 (transparent) and 1 (opaque)  # cb = fig.colorbar(colmap)  #  # ax.set_xlabel('x $[\mathrm{\AA}]$')  # ax.set_ylabel('y $[\mathrm{\AA}]$')  # ax.set_zlabel('z $[\mathrm{\AA}]$')  # plt.title(filename)  # plt.show()  # plt.close()  # pl.dump(fig,file(filename+'.pickle','w'))

    if class_name != '':
        filename = main_folder#os.path.join(main_folder, '{0}_class{1}.svg'.format(str(prefix), str(class_name)))
        plt.savefig(filename, format='svg', # before: format='pdf', also below; and {0}_class{1}.pdf!
                    dpi=1000)
    else:
        filename = main_folder#os.path.join(main_folder, '{0}_{1}.svg'.format(str(prefix), str(suffix)))
        plt.savefig(filename, format='svg', dpi=1000) 

    #logger.info("File saved to {}.".format(filename))
    plt.close()






def matrix_heatmap_2D(savefig_name,title,kernel_matrix,cmap=plt.cm.Blues,vmax=1,vmin=0):
   
    plt.xlabel('x $[\mathrm{\AA}]$')
    plt.ylabel('y $[\mathrm{\AA}]$')
    plt.title(title,  fontsize=8)
    #messed up version:
    #plt.imshow(kernel_matrix,interpolation='hanning',cmap=cmap,vmax=vmax,extent=[0,(len(kernel_matrix[0])-1)*stride_size[0],(len(kernel_matrix)-1)*stride_size[1],0])
    # If want stride label at beginning of pixel
    plt.imshow(kernel_matrix,interpolation='hanning',cmap=cmap,vmax=vmax,vmin=vmin,extent=[0,(len(kernel_matrix[0]))*stride_size[0],(len(kernel_matrix))*stride_size[1],0])    
    # If want stride label centered at pixel then do this:
    #plt.imshow(kernel_matrix,interpolation='hanning',cmap=cmap,vmax=vmax,extent=[-0.5*stride_size[0],(len(kernel_matrix[0])-1)*stride_size[0]+0.5*stride_size[0],(len(kernel_matrix)-1)*stride_size[1]+0.5*stride_size[1],-0.5*stride_size[1]])
    plt.colorbar()
    plt.show()
    plt.savefig(savefig_name)
    plt.close()




def calc_local(geometry_files, box_size, stride, configs,
               padding_ratio=None, 
               min_atoms=3, 
               adjust_box_size_by_number_of_atoms=False, min_n_atoms=100, criterion='median',
               min_atoms_spm=50, model_file=None, path_to_summary_train=None, descriptor=None,
               mc_samples=1000, plot_results=False, desc_filename=None):
    """
    geometry_files: list
        list of geometry files

    box_size: list
        list of box size values (float) to be used for each geometry file.

    stride: list
        list of list of strides to be used for each geometry file.

    padding_ratio: list, optional (default=None)
        list of 1D lists, where each element specifies the
        amount of empty space (relative to the box size, i.e.,
        taking values in [0,1]) that is appended
        at the boundaries. Choosing this to a size
        of 0.5-1.0 typically suffices.
        For the default setting, a padding of 1.0 * box_size
        is used for each spatial dimension.

    min_atoms: int, optional (default=3)
        Minimum number of atoms contained in each box
        for which a descriptor will be calculated.

    adjust_box_size_by_number_of_atoms: boolean, optional (default=False)
        Determine if the box size is automatically tuned
        such that at least 'min_n_atoms' are contained in each box.
        The keyword 'criterion' fixes if the mean or the median of
        the number of atoms is at least 'min_n_atoms'.

    min_n_atoms: int,  optional (default=100)
        If adjust_box_size_by_number_of_atoms=True, this number is
        used to increase the box size until at least min_n_atoms
        atoms are contained in each box based on the criterion fixed
        via the keyword 'criterion'.

    criterion: string, optional (default='median')
        If adjust_box_size_by_number_of_atoms = True, the box size will
        be increased until at least min_n_atoms atoms are contained either
        according to the average (criterion='average') or the
        median (criterion='median').

    model: path to h5 file, optional (default=None)
        If None, then the model used in Leitherer et. al. 2021 will be used.

    descriptor: object, optional (default=None)
        If None, the quippy SOAP descriptor will be employed automatically
        with the standard settings used in Leitherer et. al. 2021.

    mc_samples: int, optional (default=1000)
        Number of Monte Carlo sampes to calculate uncertainty estimate.

    plot_results: boolean, optional (default=False)
        Decide wheter to automatically generate svg files for visual analysis.

    """
    if not desc_filename == None:
        if not (type(desc_filename) == list or len(desc_filename)==len(geometry_files)):
            raise ValueError("If specify desc files, specifiy them as list containing at least len(geometry_files) entries.")
    
    if model_file == None:
        model_file = get_data_filename('data/nn_models/AI_SYM_Leitherer_et_al_2021.h5')
        
    if len(geometry_files) == 0:
        raise ValueError("No geometry files specified - or only passed as string and not as list.")
    
    parameters_to_check = {'stride' : stride, 'box_size' : box_size, 'padding_ratio': padding_ratio}
    if type(stride) == float or type(box_size) == float:
        raise ValueError("Please specify stride and box size as list of floats.")
    
    for key in parameters_to_check:
        parameter = parameters_to_check[key]
        print('Test parameter {}'.format(key))
        if key == 'padding_ratio':
            if parameter == None:
                parameter = [[1.0, 1.0, 1.0] for _ in range(len(geometry_files))]
                padding_ratio = parameter
        if not len(parameter) == len(geometry_files):
            raise ValueError("Parameter {} needs to be list of same length as geometry_files.".format(key))
    strides = stride
    box_sizes = box_size
    padding_ratios = padding_ratio
    """
    if not type(box_size) == list:
        box_sizes = [float(box_size)]
    else:
        box_sizes = box_size
    if not type(stride) == list:
        strides = [[float(stride), float(stride), float(stride)]]
    elif type(stride) == list:
        strides = [[_, _, _] for _ in stride]
    else:
        strides = stride
    if not type(padding_ratio) == list:
        padding_ratios = [padding_ratio]
    else:
        padding_ratios = padding_ratio
    
    if padding_ratio==None:
        padding_ratios = [[1.0, 1.0, 1.0] for _ in range(len(geometry_files))]
    """    
    
    base_folder = configs['io']['main_folder']
    structure_files = geometry_files
    
    predictions = []
    uncertainty = []
    #print(structure_files, strides, box_sizes, padding_ratios)
    geom_file_id = 0
    for structure_file, stride_size, box_size, padding_ratio in zip(structure_files, strides, box_sizes, padding_ratios):
        print('Structure file {}'.format(structure_file))
        appendix_to_folder = '_box_' + str(box_size) + '_stride_' + str(stride_size)
        
        # atoms scaling chosen automatically here to include the maximal information -> may provide that as 
        # as an option in the future.        
        atoms_scaling_cutoffs=[box_size, box_size*2, box_size*3]
        #atoms_scaling_cutoffs=[20.,30.,40.,50.]
        
        new_directory = os.path.join(base_folder, os.path.basename(structure_file)[:-4] + appendix_to_folder)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        else:
            """
            shutil.rmtree(new_directory)           #removes all the subdirectories! -> disabled for now.
            os.makedirs(new_directory)
            """
            run = 2
            while os.path.exists(new_directory + '_run_' + str(run)):
                run +=1
            new_directory = new_directory + '_run_' + str(run) 
            os.makedirs(new_directory)
        main_folder = new_directory
        
        
        # read config file
        configs_new = set_configs(main_folder=main_folder)
        #logger_new = setup_logger(configs_new, level='INFO', display_configs=False)
        # setup folder and files   - need to check for future release
        # if all of this is necessary.
        checkpoint_dir = os.path.dirname(model_file)
        checkpoint_filename = os.path.basename(model_file)
        
        dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
        conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
        results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
        
        configs_new['io']['dataset_folder'] = dataset_folder
        
    

    
        if adjust_box_size_by_number_of_atoms: 
            # In the future: refine this part: start from large box and large stride, then make it finer to get more reasonable
            # number of atoms, i.e., start with large box, also make it smaller if exceed the number of atoms!
            initial_box_size = 0
            box_size_step_size = 1
            max_spread = 10
            current_mean_natoms = 0
            current_spread = max_spread*2
            counter = 0

            start_time = time.time()
            box_size = initial_box_size
            while current_mean_natoms<min_n_atoms:# or current_spread>max_spread:
                counter +=1
                print("Iteration {}".format(counter))
                box_size += box_size_step_size
                boxes, number_of_atoms_xyz = get_boxes_from_xyz(structure_file, 
                                                                sliding_volume=[box_size, box_size, box_size], 
                                                                stride_size= [4.0, 4.0, 4.0],#[box_size/4., box_size/4., box_size/4.],
                                                                give_atom_density=True,
                                                                plot_atom_density=False, padding_ratio=[0.0,0.0,0.0])#, atom_density_filename=os.getcwd())

                current_mean_natoms = np.median(np.array(number_of_atoms_xyz).flatten())
                current_spread = np.std(np.array(number_of_atoms_xyz).flatten())
                print("Mean Natoms = {}, spread = {} ".format(current_mean_natoms, current_spread))
                
            print("Final box size = {} with natoms mean = {} and spread = {}".format(box_size, current_mean_natoms, current_spread))
            end_time = time.time()
            
            
            print("--- %s seconds ---" % (end_time - start_time))    
    
        # adjust padding ratio for slab structures
        polycrystal_structure = read(structure_file, ':', 'xyz')[0]
        positions = polycrystal_structure.positions
        for dim in range(3):
            positions_current_dim = positions[:, dim]
            extension_current_dim = abs(max(positions_current_dim)-min(positions_current_dim))
            if extension_current_dim<=box_size: # if thickness 20 A or smaller, adjust box size suitably such that only one 
            # step is takken into that direction, plus no padding is used in that direction. # TODO : only stride adjusted, still be fine prob., but actuall box size should be adjusted???
                #stride_size[dim] = round(extension_current_dim*2) # gives trouble  if extension = 0.0
                padding_ratio[dim] = 0.0
        print("Final stride = {}, final padding ratio = {}".format(stride_size, padding_ratio))   
    
    
        # Descriptor
        if descriptor == None:
            #p_b_c=False
            l_max = 6
            n_max = 9
            atom_sigma = 0.1
            cutoff = 4.0
            central_weight = 0.0
            constrain_nn_distances = False
            descriptor = quippy_SOAP_descriptor(configs=configs_new, p_b_c=False, cutoff=cutoff, l_max=l_max,
                                                n_max=n_max, atom_sigma=atom_sigma, central_weight=central_weight,
                                                average=True,average_over_permuations=False,number_averages=200,
                                                atoms_scaling='quantile_nn',atoms_scaling_cutoffs=atoms_scaling_cutoffs, extrinsic_scale_factor=1.0, 
                                                n_Z=1, Z=1, n_species=1, species_Z=1, scale_element_sensitive=True, return_binary_descriptor=True,
                                                average_binary_descriptor=True, min_atoms=min_atoms, shape_soap=316,constrain_nn_distances=constrain_nn_distances)
                                            
    
        save_file = open(os.path.join(main_folder, os.path.basename(structure_file)[:-4]+'_log_file.txt'),'w') 
        # comment if you have already calculated the descriptor for the .xyz file
        desc_filename_to_load = None
        if not desc_filename == None:
            desc_filename_to_load = desc_filename[geom_file_id]
            geom_file_id += 1
        
        start = time.time()
        path_to_x_test, path_to_y_test, path_to_summary_test, path_to_strided_pattern_pos = make_strided_pattern_matching_dataset(
            polycrystal_file=structure_file, descriptor=descriptor, desc_metadata='SOAP_descriptor',
            configs=configs_new, operations_on_structure=None, stride_size=stride_size, box_size=box_size,
            init_sliding_volume=None, desc_file=desc_filename_to_load, desc_only=False, show_plot_lengths=False,
            desc_file_suffix_name='', nb_jobs=16, padding_ratio=padding_ratio, min_nb_atoms=min_atoms_spm)#min_atoms)
        end = time.time()
        ex_time = str(end-start)
        print('Execution time descriptor calculation: '+ex_time)
        #print(path_to_x_test)
        #print(path_to_y_test)
        #print(path_to_summary_test)
        #print(path_to_strided_pattern_pos)
        save_file.write('Runtime crystal'+structure_file+' '+ex_time) 


        # copy soap information into dataset folder (need to find more elegant way in the future)
        #shift_training_data_to_different_path(configs_new['io']['dataset_folder'])
        configs_new['io']['polycrystal_file'] = os.path.basename(structure_file)

        start = time.time()
        get_classification_map(configs_new, path_to_x_test, path_to_y_test, path_to_summary_test, path_to_strided_pattern_pos, checkpoint_dir, checkpoint_filename=checkpoint_filename,
                               mc_samples=mc_samples, interpolation='none', results_file=None, calc_uncertainty=True,
                               conf_matrix_file=conf_matrix_file, train_set_name='soap_pristine_data',
                               cmap_uncertainty='hot', interpolation_uncertainty='none', plot_results=plot_results, path_to_summary_train=path_to_summary_train)
        end = time.time()
        prediction_str = 'Time for predicting '+str(end-start)+' s \n'
        save_file.write(prediction_str)
        save_file.write('Box size '+str(box_size)+', stride_size '+str(stride_size)+' padding_ratio '+str(padding_ratio)+' min_atoms for quippy: '+str(min_atoms)+' minatoms SPM '+str(min_atoms_spm)+' cutoff_for_scaling '+str(atoms_scaling_cutoffs))
        save_file.close()
        
        # load and append predictions and uncertainty
        prediction = np.load(os.path.join(configs_new['io']['results_folder'],
                             configs_new['io']['polycrystal_file'] + '_probabilities.npy'))
        predictions.append(prediction)
        
        uncertainty_dict = {'mutual_information': [], 'variation_ratio': [], 'predictive_entropy': []}
        for key in uncertainty_dict:
            uncertainty_ = np.load(os.path.join(configs_new['io']['results_folder'],
                    configs_new['io']['polycrystal_file'] + '_' + key + '.npy'))
            uncertainty_dict[key] = uncertainty_
        uncertainty.append(uncertainty_dict)
        
        
        print('Clean tmp folder')
        clean_folder(configs_new['io']['tmp_folder'], endings_to_delete=(".png", ".npy", "_target.json","_aims.in", "_ase_atoms_info.pkl", "_ase_atoms.json", "_coord.in"))    
        
        
    return predictions, uncertainty
     
     
"""
def plot_stuff():

        if not do_3D:
            continue

        # plot
        # Load info about classes
        with open(os.path.join(dataset_folder,'soap_pristine_data_summary.json')) as data_file:    
            data = json.load(data_file)
        numerical_labels = data["data"][0]['classes']
        numerical_to_text_label = dict(zip(range(len(numerical_labels)), numerical_labels))    
        text_to_numerical_label = dict(zip(numerical_labels, range(len(numerical_labels)))) 
        number_labels = [text_to_numerical_label[txtlabel] for txtlabel in numerical_labels]

        l12_label = text_to_numerical_label['L12_Cu3Au']
        fcc_label = text_to_numerical_label['fcc_Cu_A_cF4_225_a']
        l10label = text_to_numerical_label['L10_CuAu']

        filenames = []
        filenames.append('Probability_prob_class'+str(l12_label)+'.npy')
        filenames.append('Probability_prob_class'+str(fcc_label)+'.npy')
        #filenames.append('Proto '+numerical_to_text_label[bcc_label]+' Probability_prob_class'+str(bcc_label)+'.npy')
        #filenames.append('Proto '+numerical_to_text_label[diam_label]+' Probability_prob_class'+str(diam_label)+'.npy')
        #filenames.append('Proto '+numerical_to_text_label[hcp_label]+' Probability_prob_class'+str(hcp_label)+'.npy')
        filenames.append('Probability_prob_class'+str(l10label)+'.npy')
        filenames.append('Uncertainty (predictive_entropy)_uncertainty_class.npy')
        filenames.append('Uncertainty (mutual_information)_uncertainty_class.npy')
        filenames.append('Uncertainty (variation_ratio)_uncertainty_class.npy')
        filenames = [os.path.join(main_folder, item) for item in filenames]


        # 2D plot:
        
        with open(path_to_strided_pattern_pos, 'rb') as input_spm_pos:
            strided_pattern_pos = pickle.load(input_spm_pos)
        class_plot_pos = np.asarray(strided_pattern_pos)
        (z_max, y_max, x_max) = np.amax(class_plot_pos, axis=0) + 1
        
        df_positions = pd.DataFrame(data=class_plot_pos,
                                    columns=['strided_pattern_positions_z', 'strided_pattern_positions_y',
                                             'strided_pattern_positions_x'])
        
        df_positions_sorted = df_positions.sort_values(
            ['strided_pattern_positions_z', 'strided_pattern_positions_y', 'strided_pattern_positions_x'], ascending=True)
        
        for filename in filenames:
            
            if filename=='Normalized uncertainty':
                continue
                epsilon = 1e-20 # avoid dividing by zero
                mutual_info = np.load(folder_name +'/'+'Uncertainty (mutual_information)_uncertainty_class.npy')
                mutual_info = np.nan_to_num(mutual_info)
                predictive_entropy = np.load(folder_name +'/'+'Uncertainty (predictive_entropy)_uncertainty_class.npy')
                predictive_entropy = np.nan_to_num(predictive_entropy)
                
                prob = np.divide(mutual_info , predictive_entropy + epsilon)
                prob = np.nan_to_num(prob)
                
            
            else:
                prob = np.load(filename)
                prob = np.nan_to_num(prob)
            
            for _2dheatmap,xyz in zip(prob, range(z_max)):#zip(prob,df_positions_sorted.values):
                identifier = filename.split('/')[-1] # gives eg 'Uncertainty (variation_ratio)_uncertainty_class.npy'
                identifier = identifier[:-4]
                # get z position: we start at min        
                z = xyz*stride_size[2]
                if 'uncertainty' in filename:
                    cmap = 'hot'
                    vmax = max(np.array(_2dheatmap).flatten())
                    vmin = min(np.array(_2dheatmap).flatten())
                else:
                    cmap = 'viridis'
                    vmax = 1.0
                    vmin = 0.0
                #savefilename = os.path.join(folder_name,filename[:-4]+'_z_'+str(z))
                #matrix_heatmap_2D(savefig_name=savefilename+'.svg', title= 'z = '+str(z)+'  '+identifier+'\n'+class_info, kernel_matrix=_2dheatmap,
                #                          cmap=cmap,vmax=vmax,vmin=vmin)
                title = 'z = '+str(z)+'  '+identifier#+'\n'+class_info
                save_filename = identifier+'_z_'+str(z)+'.svg'
                plot_prediction_heatmaps_(_2dheatmap, title=title, class_name='', prefix='prob',
                                         main_folder=os.path.join(main_folder, save_filename), cmap=cmap, color_nan='lightgrey',
                                         interpolation='none', vmin=vmin, vmax=vmax) # added vmin, vmax here


        print('Clean tmp folder')
        clean_folder(configs['io']['tmp_folder'], endings_to_delete=(".png", ".npy", "_target.json","_aims.in", "_ase_atoms_info.pkl", "_ase_atoms.json", "_coord.in"))
"""