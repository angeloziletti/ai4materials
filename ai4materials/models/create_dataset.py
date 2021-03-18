
import matplotlib
import matplotlib.pyplot as plt

import os.path
import sys
import ase
from ase.io import write, read

import numpy as np

from copy import deepcopy

from ai4materials.descriptors.ft_soap_descriptor import FT_SOAP_harmonics
from ai4materials.descriptors.quippy_soap_descriptor import quippy_SOAP_descriptor

from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger

from ai4materials.utils.utils_crystals import create_vacancies, create_supercell
from ai4materials.utils.utils_crystals import random_displace_atoms


from ai4materials.wrappers import calc_descriptor_in_memory
from ai4materials.utils.utils_config import get_data_filename


import logging
logger = logging.getLogger('ai4materials')

import itertools

def calc_dataset(configs, prototypes, nb_jobs=-1, min_nb_atoms=100, periodic_boundary_conditions=[True, False], 
                 descriptor_params_to_vary=None, **kwargs):
    """
    UNFINISHED
    Given set of prototypes, calculate all possible descriptors as specified in 'descriptor_params_to_vary'.

    """                 
                     
    # function arguments
    geometry_files = ['bcc.in', 'fcc.in']
    descriptors_and_parameters = [{'descriptor': quippy_SOAP_descriptor, 'fixed_parameters': {}, 'varying_parameters': {'cutoff': [2.0, 3.0], 'atom_sigma': [0.1, 0.2]}} ]#{'cutoff': np.linspace(3.0, 5.0, 11), 'atom_sigma': np.linspace(0.08, 1.2, 3)} }]
    file_format = 'aims'
    
    
    all_descriptor_names = [d['descriptor'].__name__ for d in descriptors_and_parameters]
    results_dict = {descriptor_name: defaultdict() for descriptor_name in all_descriptor_names}
    
    for descriptor_parameter_info_dict in descriptors_and_parameters:
        descriptor = descriptor_parameter_info_dict['descriptor']
        fixed_parameters = descriptor_parameter_info_dict['fixed_parameters']
        
        structures = []
        for geometry_file in geometry_files:
            ase_atoms = read(geometry_file, ':', format=file_format)[0]
            structures.append(ase_atoms)
            
        parameters_dict = descriptor_parameter_info_dict['varying_parameters']
        parameters_list = [parameters_dict[_] for _ in parameters_dict]
        parameters_list_to_string = [_ for _ in parameters_dict]
        all_combinations = itertools.product(*parameters_list)

        for combination in all_combinations:
            combination_dict = {descritpor_parameter: _  for descritpor_parameter,_ in zip(parameters_list_to_string, list(combination))}
            descriptor = deepcopy(descriptor_parameter_info_dict['descriptor'])
            all_arguments = {}
            all_arguments.update(combination_dict)
            all_arguments.update(fixed_parameters)
            descriptor = descriptor(**all_arguments)

def calc_SOAP_dataset_from_geofiles(configs, geometry_files, **kwargs):
    material_type = 'Unspecified'
    directory_name = os.path.dirname(geometry_files[0])
    
    material_type_path = os.path.join(directory_name, material_type)
    if not os.path.exists(material_type_path):
        os.makedirs(material_type_path)
    for geo_file in geometry_files:
        name_wo_ending = os.path.basename(geo_file).split('.')[0]
        name_w_ending = os.path.basename(geo_file)
        structural_class_path = os.path.join(material_type_path, name_wo_ending)
        if not os.path.exists(structural_class_path):
            os.makedirs(structural_class_path)
        atoms = load(geo_file)
        write(os.path.join(structural_class_path, name_w_ending), atoms)
    calc_SOAP_dataset(configs, prototypes_path=directory_name)

def calc_SOAP_dataset(configs, nb_jobs=6, prototypes_path=None, iterations=1,
                               min_nb_atoms=100, cubic_shape_1=120, cubic_shape_2=240,
                               periodic_boundary_conditions=[True], cutoff_range=[3.0, 4.0, 5.0],
                               atom_sigma_range = np.array([0.08,0.1,0.12]),
                               cutoff_range_defective=[4.0], atom_sigma_range_defective=[0.1],
                               n_max=9, l_max=6, central_weight=0.0, scale_element_sensitive=True,
                               vacancy_ratios = np.array([0.01, 0.02]),
                               displacement_ratios = np.array([0.001, 0.002, 0.006]),
                               extrinsic_scaling_factors=[1.0], cutoff_for_scaling=[10., 20.],
                               return_binary_descriptor=True, average_binary_descriptor=True, average_harmonics=True):
    """

    configs: dict
        Configuration file, necessary for getting main folder etc.
    nb_jobs: int, default=-1
        # CPUs employed for the calculation
    prototypes_path: string, default set to Leitherer et.al. prototypes. To calculate a new dataset, the following folder structure has to 
        be created:
        PROTOTYPES/
            Material_type_1 (e.g., Elemental_solids etc.)
                Structural class (e.g. 'fcc')
                    Prototypical structure ('fcc.in')
            Material_type_2
            .
            .
            .
    iterations: int, optional
        if 0, then no defective structures will be calculated. If >=1, then >=1 iteration of random noise will be applied to
        each structure in the prototype (while for non-periodic structures, supercell structures are calculated automatically.)
    periodic_boundary_conditions: list
        type of pbc applied to each structure. For pbc=False, supercells are calculated automatically with the 
        smallest size being determined by min_nb_atoms
    
                
    
    """

    vac_and_displ = zip(vacancy_ratios,displacement_ratios)    
    
    
    if prototypes_path == None:
        prototypes_path = get_data_filename('data/PROTOTYPES')
    

    appendix_to_descfiles = ''
    

    
    # element ordering
    #orders = ['12','21'] #12 ordering: smallest atomic number first. 21 ordering: biggest atomic number first

    # Descriptors to be calculated
    descriptors_to_calculate = ['soap']   
    
    # Options for creating supercells, used when introducing defects below
    target_replicas = (1, 1, 1)
    kwargs = dict(create_replicas_by='user-defined', 
    #              create_replicas_by='nb_atoms',
                  max_diff_nb_atoms=100, 
                  random_rotation_before=False, 
                  target_nb_atoms=128,
                  random_rotation=False, 
                  cell_type=None,
                  optimal_supercell=False, 
                  target_replicas=target_replicas)
    
    
    ########################################
    # CALCULATION
    ######################################## 

    # Compute descriptors for all kinds of datasets:
    logger.info('Load prototypes specified in prototype path.')
    # get different datasets
    datasets = os.listdir(prototypes_path)
    for p_b_c in periodic_boundary_conditions:
        for dataset in datasets:        

            prototypes_basedir = os.path.join(prototypes_path,dataset)
            crystal_structures = os.listdir(prototypes_basedir)
            
            # LOAD PROTOTYPES
            # go trhough folder, get all txt files, load all of protos, give them labels according to their filename which will identify them uniquely  
            prototypes = []
            for crystal_structure in crystal_structures:
                all_files = os.listdir(os.path.join(prototypes_basedir, crystal_structure))
                all_files = [x for x in all_files if x[-4:]=='.txt'] # only consider txt files
                for i,filename in enumerate(all_files):
                    prototype = ase.io.read(os.path.join(prototypes_basedir,crystal_structure,filename), ':', 'aims')[0]
                    prototype.info['idx'] = i
                    prototype.info['label'] = appendix_to_descfiles+'_'+filename[:-4]+'_pbc_'+str(p_b_c) # label for assigning unique name to all files in the tar files
                    prototype.info['crystal_structure'] = crystal_structure # THIS WILL BE USED AS TARGET FOR TRAINING
                    prototype.info['dataset'] = dataset
                    
                    if prototype.cell.flatten().any() == 0.0:
                        logger.info('Prototype {} has no cell, skipped'.format(prototype.info['label']))
                        continue # skipt eg Nanotubes
                    # Elongate cell such that do not run into trouble for pbc=True
                    if dataset=='2D_materials':
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
                    if dataset=='Nanotubes' or prototype.cell.flatten().any()==0.0:
                        prototype.set_pbc(False)
                    elif dataset=='2D_materials':
                        prototype.set_pbc([p_b_c,p_b_c,p_b_c])#False])
                    else:
                        prototype.set_pbc(p_b_c)
                    prototypes.append(prototype)
            
            # GET PRISTINE AND DEFECTIVE SUPERCELL STRUCTURES
            prototypes_pristine = []
            target_list_pristine = []

            # Initialize defective dict
            descriptor_dict_defective_structures =  {'vacancies': {x: [] for x in vacancy_ratios} ,
                                                     'displacements': {x: [] for x in displacement_ratios},
                                                     'vac_and_displ': {x: [] for x in vac_and_displ} }
                                                     
            target_dict_defective_structures =  {'vacancies': {x: [] for x in vacancy_ratios} ,
                                                     'displacements': {x: [] for x in displacement_ratios},
                                                     'vac_and_displ': {x: [] for x in vac_and_displ} }
            
            # Define structures
            for prototype in prototypes:
                # PRISTINE
                if p_b_c==True:
                    supercell_structure = prototype #*(3,3,1)
                    prototypes_pristine.append(supercell_structure)
                    target = supercell_structure.info['crystal_structure']
                    target_list_pristine.append(target)
    
                # pbc_ used to recover pbc, set to False for defective supercell structure calculations
                # to save calculation time
                if prototype.info['dataset']=='Nanotubes':
                    pbc_ = False
                elif prototype.info['dataset']=='2D_materials':
                    pbc_ = [p_b_c, p_b_c, p_b_c]#False]
                else:
                    pbc_ = [p_b_c, p_b_c, p_b_c]#was set to False bef0re ---> might have given displaced structures with wrong pbc!!!
                
                if p_b_c==True:
                    supercells_array = prototype.info['supercells']#[prototype.info['supercells'][1]]
                elif p_b_c==False:
                    supercells_array = prototype.info['supercells']
                for supercell in supercells_array:#prototype.info['supercells']:
                    # PRISTINE
                    logger.info('Get supercells for pristine structures.')
                    if supercell=='Nanotubes':
                        supercell_structure = prototype
                        #if p_b_c==True: # skip Nanotubes if pbc True
                        #    continue
                    elif supercell=='cubic_shape_1':
                        supercell_structure = create_supercell(prototype,target_nb_atoms=cubic_shape_1,optimal_supercell=True)
                    elif supercell=='cubic_shape_2':
                        supercell_structure = create_supercell(prototype,target_nb_atoms=cubic_shape_2,optimal_supercell=True)
                    else:
                        supercell_structure = prototype*supercell
                    supercell_structure.info['label']+='_supercell_'+str(supercell_structure.get_number_of_atoms())+'_'+str(supercell)
                    if p_b_c==False:    
                        prototypes_pristine.append(supercell_structure)
                        #target used for pristine and defective structures
                        target = supercell_structure.info['crystal_structure']
                        target_list_pristine.append(target)
                    
                    # DEFECTIVE
                    logger.info('Get supercells for defective structures.')
                    #if len(supercell_structure)<100:
                    #    continue # skip structures with less than 100 atoms, otherwise get strange structures

                    supercell_structure.set_pbc(False)
                    for it in range(iterations):
                        # HACK - momentarily uncomment vac and displ
                        
                        # vacancies
                        for vacancy_ratio in vacancy_ratios:
                            vacancy_structure = create_vacancies(supercell_structure,target_vacancy_ratio=vacancy_ratio, **kwargs)
                            vacancy_structure.info['label'] = supercell_structure.info['label']+'_'+str(vacancy_structure.get_number_of_atoms())+'_it_'+str(it)+'_vac_ratio_'+str(round(vacancy_ratio,3))
                            
                            vacancy_structure.set_pbc(pbc_)
                            descriptor_dict_defective_structures['vacancies'][vacancy_ratio].append(vacancy_structure)
                            target_dict_defective_structures['vacancies'][vacancy_ratio].append(target)
                        # displacements
                        for displacement_ratio in displacement_ratios:
                            displacement_structure = random_displace_atoms(atoms=supercell_structure,noise_distribution='uniform_scaled',displacement_scaled=displacement_ratio,**kwargs)
                            displacement_structure.info['label'] = supercell_structure.info['label']+'_it_'+str(it)+'_displ_ratio_'+str(round(displacement_ratio,3))
                            
                            displacement_structure.set_pbc(pbc_)
                            descriptor_dict_defective_structures['displacements'][displacement_ratio].append(displacement_structure)
                            target_dict_defective_structures['displacements'][displacement_ratio].append(target)
                        
                        # vacancies and displacements
                        for vac_displ_ratio in vac_and_displ:
                            vac_ratio = vac_displ_ratio[0]
                            displ_ratio = vac_displ_ratio[1]
                            vacancy_structure = create_vacancies(supercell_structure,target_vacancy_ratio=vac_ratio, **kwargs)
                            vac_and_displ_structure = random_displace_atoms(atoms=vacancy_structure,noise_distribution='uniform_scaled',displacement_scaled=displ_ratio,**kwargs)
                            vac_and_displ_structure.info['label'] = supercell_structure.info['label']+'_it_'+str(it)+'_'+str(vacancy_structure.get_number_of_atoms())+'_displ_ratio_'+str(round(displ_ratio,3))+'_vac_ratio_'+str(round(vac_ratio,3))
                            
                            vac_and_displ_structure.set_pbc(pbc_)
                            descriptor_dict_defective_structures['vac_and_displ'][vac_displ_ratio].append(vac_and_displ_structure)
                            target_dict_defective_structures['vac_and_displ'][vac_displ_ratio].append(target)
                    supercell_structure.set_pbc(pbc_)
                    
                        
            # Compute pristine descriptors
            logger.info('Calculate descriptor for different parameters. Start with pristine structures.')
            for descriptor_to_calculate in descriptors_to_calculate:
                for cutoff in cutoff_range:
                    for atom_sigma in atom_sigma_range:

                        ##############
                        # PRISTINE
                        ##############
                        
                        for extrinsic_scaling_factor in extrinsic_scaling_factors:
                            if descriptor_to_calculate=='soap':
                                descriptor = quippy_SOAP_descriptor(configs=configs, p_b_c=False, cutoff=cutoff, l_max=l_max,
                                                                    n_max=n_max, atom_sigma=atom_sigma, central_weight=central_weight,
                                                                    average=True,average_over_permuations=False,number_averages=200,
                                                                    atoms_scaling='quantile_nn',atoms_scaling_cutoffs=cutoff_for_scaling, extrinsic_scale_factor=extrinsic_scaling_factor, 
                                                                    n_Z=1, Z=1, n_species=1, species_Z=1, scale_element_sensitive=scale_element_sensitive, return_binary_descriptor=return_binary_descriptor,
                                                                    average_binary_descriptor=average_binary_descriptor)
                            elif descriptor_to_calculate=='ft_soap':
                                descriptor = FT_SOAP_harmonics(configs=configs,p_b_c=False,cutoff=cutoff,l_max=l_max,n_max=n_max,atom_sigma=atom_sigma,central_weight=central_weight,
                                                               average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',
                                                               atoms_scaling_cutoffs=cutoff_for_scaling,number_of_harmonics=158,real_or_imag_or_spec='power_spectrum',
                                                               discard_full_spectrum=True, extrinsic_scale_factor=extrinsic_scaling_factor,
                                                               n_Z=1, Z=1, n_species=1, species_Z=1, scale_element_sensitive=scale_element_sensitive, return_binary_descriptor=return_binary_descriptor,
                                                               unit_norm=False, average_harmonics = average_harmonics)
                            else:
                                raise NotImplementedError("Given descriptor not implemented.")
                                
                            # Uncomment if want info about cutoff and sigma in each plot:
                            original_labels_of_atoms=[] # used for recovering original label in next run, otherwise will accumulate               
                            cutoff_sigma_info='_cutoff_'+str(round(cutoff,4))+'_sigma_'+str(round(atom_sigma,4))+'_exsf_'+str(extrinsic_scaling_factor)
                            for structure in prototypes_pristine:
                                original_labels_of_atoms.append(structure.info['label'])
                                structure.info['label']+=cutoff_sigma_info
                
                            
                            desc_file_name=appendix_to_descfiles+dataset+'_'+descriptor_to_calculate+'_pristine_cutoff_'+str(round(cutoff,4))+'_atom_sigma_'+str(round(atom_sigma,4))+'_exsf_'+str(extrinsic_scaling_factor)+'_pbc_'+str(p_b_c)+'.tar.gz'
                            desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs, ase_atoms_list=prototypes_pristine,
                                                             tmp_folder=configs['io']['tmp_folder'],
                                                             desc_folder=configs['io']['desc_folder'],
                                                             desc_file=desc_file_name,
                                                             format_geometry='`aims',
                                                             nb_jobs=nb_jobs,
                                                             target_list = target_list_pristine)
                            # Reset label, only needed if add cutoff and sigma to .info of structures                              
                            for structure,label in zip(prototypes_pristine,original_labels_of_atoms):
                                structure.info['label']=label
            logger.info('Calculate descriptor for different parameters. Continue with defective structures.')                    
            # Compute descriptors
            for descriptor_to_calculate in descriptors_to_calculate:
                for cutoff in cutoff_range_defective:
                    for atom_sigma in atom_sigma_range_defective:
                        ##################
                        # DEFECTIVE 
                        ##################
                        if iterations==0:
                            continue
                        # Define descriptor object (same for all defective structures since use extr.sf of 1 for them.)
                        if descriptor_to_calculate=='soap':
                            descriptor = quippy_SOAP_descriptor(configs=configs, p_b_c=False, cutoff=cutoff, l_max=l_max,
                                                                n_max=n_max, atom_sigma=atom_sigma, central_weight=central_weight,
                                                                average=True,average_over_permuations=False,number_averages=200,
                                                                atoms_scaling='quantile_nn',atoms_scaling_cutoffs=cutoff_for_scaling, extrinsic_scale_factor=1.0, #exsf set to one for defective structures!
                                                                n_Z=1, Z=1, n_species=1, species_Z=1, scale_element_sensitive=scale_element_sensitive, return_binary_descriptor=return_binary_descriptor,
                                                                average_binary_descriptor=average_binary_descriptor)
                        elif descriptor_to_calculate=='ft_soap':
                            descriptor = FT_SOAP_harmonics(configs=configs,p_b_c=False,cutoff=cutoff,l_max=l_max,n_max=n_max,atom_sigma=atom_sigma,central_weight=central_weight,
                                                           average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',
                                                           atoms_scaling_cutoffs=cutoff_for_scaling,number_of_harmonics=158,real_or_imag_or_spec='power_spectrum',
                                                           discard_full_spectrum=True, extrinsic_scale_factor=1.0,
                                                           n_Z=1, Z=1, n_species=1, species_Z=1, scale_element_sensitive=scale_element_sensitive, return_binary_descriptor=return_binary_descriptor,
                                                           unit_norm=False, average_harmonics = average_harmonics)
                        else:
                            raise NotImplementedError("Given descriptor not implemented.")
                        
                        
                        for defect_type in descriptor_dict_defective_structures:
                            # defect_type is either vacancies, displacements or vac_and_displ
                            #if not defect_type=='vac_and_displ':
                            #    continue    
                            for defect_ratio in descriptor_dict_defective_structures[defect_type]:
                                
                                defective_prototypes = descriptor_dict_defective_structures[defect_type][defect_ratio]
                                defective_targets = target_dict_defective_structures[defect_type][defect_ratio]
                                
                                # Uncomment if want info about cutoff and sigma in each plot
                                original_labels_of_atoms=[] # used for recovering original label in next run, otherwise will accumulate               
                                cutoff_sigma_info='_cutoff_'+str(round(cutoff,4))+'_sigma_'+str(round(atom_sigma,4))
                                for structure in defective_prototypes:
                                    original_labels_of_atoms.append(structure.info['label'])
                                    structure.info['label']+=cutoff_sigma_info
                    
                                
                                desc_file_name=appendix_to_descfiles+dataset+'_'+descriptor_to_calculate+'_'+defect_type+'_'+str(defect_ratio)+'_cutoff_'+str(round(cutoff,4))+'_atom_sigma_'+str(round(atom_sigma,4))+'_pbc_'+str(p_b_c)+'.tar.gz'
                                desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs, ase_atoms_list=defective_prototypes,
                                                                 tmp_folder=configs['io']['tmp_folder'],
                                                                 desc_folder=configs['io']['desc_folder'],
                                                                 desc_file=desc_file_name,
                                                                 format_geometry='`aims',
                                                                 nb_jobs=nb_jobs,
                                                                 target_list = defective_targets)
                                 
                                                               
                                # Reset label, only needed if add cutoff and sigma to .info of structures                              
                                for structure,label in zip(defective_prototypes,original_labels_of_atoms):
                                    structure.info['label']=label
                        
                            
                        
        
    