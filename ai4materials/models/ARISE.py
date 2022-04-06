import os
from ai4materials.models.cnn_polycrystals import predict_with_uncertainty
from ase.io import read
from keras.models import load_model
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger
from ai4materials.descriptors.quippy_soap_descriptor import quippy_SOAP_descriptor
from ai4materials.utils.utils_config import get_data_filename
import numpy as np
import json
from ai4materials.models.spm_prediction_tools import calc_local
import pickle

import logging
logger = logging.getLogger('ai4materials')

from copy import deepcopy

import keras

def analyze_predictions(geometry_files, predictions, uncertainties,
                        numerical_to_text_label=None, top_n=3,
                        uncertainty_quantifiers=['mutual_information'], return_predicted_prototypes=False):
    if numerical_to_text_label == None:
        class_info_path = get_data_filename('data/nn_models/ARISE_class_info.json')
        with open(class_info_path) as file_name:
            data = json.load(file_name)
        class_labels = data["data"][0]['classes']
        numerical_to_text_label = dict(zip(range(len(class_labels)), class_labels))    
        text_to_numerical_label = dict(zip(class_labels, range(len(class_labels))))
    
    # uncertainties come as {'mutual_information': [u_file_1, u_file_2, ...], '': [], ...}
    geo_file_idx = 0 
    top_n_labels_list = []
    for geometry_file, prediction in zip(geometry_files, predictions):
        print('For geometry file {}'.format(geometry_file))
        top_n_indices = np.argsort(prediction.flatten())[-top_n:]
        top_n_indices = np.flip(top_n_indices)
        top_n_probabilities = prediction.flatten()[top_n_indices]
        top_n_labels = [numerical_to_text_label[x_] for x_ in top_n_indices]
        top_n_labels_list.append(top_n_labels)
        for idx in range(top_n):
            #print(idx)
            print('{}. Predicted prototype {} with classification probability {:.4f}'.format(idx+1, top_n_labels[idx], top_n_probabilities[idx]))
        uncertainty_string = ''
        for u_type in uncertainty_quantifiers:
            uncertainty_string += u_type + ' = ' + str(round(uncertainties[u_type][geo_file_idx], 4)) + ' '
        print('Uncertainty: {}'.format(uncertainty_string))
        geo_file_idx += 1
    if return_predicted_prototypes:
        return top_n_labels_list


def preparations(main_folder, p_b_c=False, l_max=6,
                 n_max=9, atom_sigma=0.1, cutoff=4.0,
                 central_weight=0.0,
                 atoms_scaling_cutoffs=[10., 20., 30., 40., 50.], 
                 min_atoms=1, constrain_nn_distances=False, logger=None, configs=None, model=None, descriptor=None):
    # read config file
    """
    if logger == None:
        if main_folder is None:
            main_folder = os.getcwd()
        configs = set_configs(main_folder=main_folder)
        logger = setup_logger(configs, level='INFO', display_configs=False)
    else:
        logger = logging.getLogger('ai4materials')
    """

    # set up descriptor
    # Descriptor
    if descriptor == None:
        p_b_c = p_b_c
        l_max = l_max
        n_max = n_max
        atom_sigma = atom_sigma
        cutoff = cutoff
        central_weight = central_weight
        descriptor = quippy_SOAP_descriptor(configs=configs, p_b_c=False,
                                            cutoff=cutoff, l_max=l_max,
                                            n_max=n_max, atom_sigma=atom_sigma,
                                            central_weight=central_weight,
                                            average=True,
                                            average_over_permuations=False,
                                            number_averages=200,
                                            atoms_scaling='quantile_nn',
                                            atoms_scaling_cutoffs=atoms_scaling_cutoffs,
                                            extrinsic_scale_factor=1.0, 
                                            n_Z=1, Z=1, n_species=1, 
                                            species_Z=1, scale_element_sensitive=True,
                                            return_binary_descriptor=True,
                                            average_binary_descriptor=True, min_atoms=min_atoms,
                                            shape_soap=316, constrain_nn_distances=constrain_nn_distances)
                                        
    # load model
    if model == None:
        model_file = get_data_filename('data/nn_models/ARISE_Leitherer_et_al_2021.h5')
        model = load_model(model_file)
    
    return descriptor, model


def global_(geometry_files, main_folder=None, n_iter=1000, configs=None,
            model=None, format_='aims', descriptor=None,
            descriptors=None, save_descriptors=False,
            save_path_descriptors=None, **kwargs): #, logger=None):
    # TODO: change functionality such that at least model prediction is done in parallel (maybe descriptor calculation in parallel as optional since it will create new folders)
    if not descriptor == None:
        specified_descriptor = deepcopy(descriptor)
        descriptor, model = preparations(main_folder, configs=configs, model=model, **kwargs) #, logger)
        descriptor = specified_descriptor
    else:
        descriptor, model = preparations(main_folder, configs=configs, model=model, **kwargs)
    input_shape_from_model = model.layers[0].get_input_at(0).get_shape().as_list()[1:]
    target_shape = tuple([-1] + input_shape_from_model)

    if not type(descriptors) == type(None):
        soap_descriptors = descriptors
    else:
        soap_descriptors = []
        structures = []
    
        for geometry_file in geometry_files:
            # read structure into ase object
            structure = read(geometry_file, ':', format=format_)[0] # TODO more general formatting
                             #format=geometry_file.split('.')[-1])
    
            structures.append(structures)
            # calculate descriptor
            soap_desc = descriptor.calculate(structure).info['descriptor']['SOAP_descriptor']
            soap_descriptors.append(soap_desc)
    # calculate predictions
    data = np.reshape(soap_descriptors, target_shape)
    prediction, uncertainty = predict_with_uncertainty(data, model=model, model_type='classification', n_iter=n_iter)
    
    if save_descriptors:
        if save_path_descriptors == None:
            base_paths = [os.path.basename(_) for _ in geometry_files]
            splitted_base_paths = [_.split('.')[0] for _ in base_paths]
            geo_file_string = '_'.join(splitted_base_paths)
            save_path_descriptors = os.path.join(os.getcwd(), geo_file_string + '.npy')
        np.save(save_path_descriptors, soap_descriptors)

    # TODO save predictions, uncertainty, structures, descriptors into ase database file
    return prediction, uncertainty
    
    
def local(geometry_files, stride, box_size, configs, n_iter=1000, main_folder=None,
          descriptor=None, model=None, format_='aims',
          desc_filename=None, nb_jobs=-1, **kwargs):
    
    # read config file
    """
    if main_folder == None:
        main_folder = os.getcw()
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)
    """
    
    predictions, uncertainty = calc_local(geometry_files, box_size, stride, configs,
                                          descriptor=descriptor, model_file=model,
                                          desc_filename=desc_filename, nb_jobs=nb_jobs, **kwargs)
    
    return predictions, uncertainty
    
def analyze(geometry_filenames, mode='global', training_info=None, stride=None,
            box_size=None, configs=None, descriptor=None, model=None,
            format_=None, descriptors=None, save_descriptors=False, 
            save_path_descriptors=None, nb_jobs=-1, **kwargs):
    """
    Apply ARISE to given list of geometry files.
    
    This function is key to reproduce the single- and polycrystalline predictions in:
    
    [1] A. Leitherer, A. Ziletti, and L.M. Ghiringhelli, Nat. Commun. 12, 6234 (2021). 
    DOI:  https://doi.org/10.1038/s41467-021-26511-5
    
    Parameters:
    
    gometry_filenames: list
        list of geometry files to be analyzed.
    
    mode: str (default='global')
        If 'global', a global descriptor will be calculated and a global label (plus uncertainty) predicted.
        If 'local', the strided pattern matching algorithm introduced in [1] is applied.    

    stride: float (default=None)
        Step size in strided pattern matching algorithm. Only relevant if mode='local'. 
        If no value is specified, a stride of 4 Angstroem in each direction, for each of the geometry files
        is used.

    box_size: float (default=None)
        Size of the box employed in strided pattern matching algorithm. Only relevant if mode='local'.
        If no value is specified, a box size of 16 Angstroem is used, for each of the geometry files.

    configs: object (default=None)
        configuration object, defining folder structure. For more details, please have a look at the function set_configs from ai4materials.utils.utils_config
        
    descriptor: ai4materials descriptor object (default=None)
        If None, the SOAP descriptor as implemented in the quippy package (see ai4materials.descritpors.quippy_soap_descriptor)
        with the standard settings employed in [1] will be used.
        
    model: str, (default=None)
        If None, the model of [1] will be automatically loaded. Otherwise the path to the model h5 file needs to be specified alongside
        information on the training set (in particular, the relation between integer class labels and 
        class labels).
        
    training_info: path to dict (default=None)
        Information on the realtion between int labels and structure labels. If model=None, training information
        of [1] will be loaded regardless of this keyword. If model not None, 
        then specification of training_info is mandatory. The structure of this dictionary 
        is defined as dict = {'data': ['nb_classes': 108, 
        'classes': [text label class 0, text label class 1, ... ie ordered class labels]]}

    format_: str, optional (default=None)
        format of geometry files. If not specified, the input files are assumed to have aims format in case of
        global mode, and xyz format in case of local mode.

    descriptors: path to desc or numpy array, optional (default=None)
        If mode=local, then this must be a path to a desc file containing the descriptors.
        If mode=global, then this must be a numpy array containing the descriptors. 

    save_descriptors: bool, optional (default=False)
        Decides whether to save calculated descriptors into specified savepath or not (only for mode=local).

    save_path_descriptors: str, optional (default=None)
        path into which descriptors are saved (for mode=global)
    """
    
    if not model == None:
        if training_info == None:
            raise ValueError("No information on the relation between int and str class labels is provided.")
    #if not (type(model) == str or type(model)==keras.engine.training.Model):
    #    raise NotImplementedError("Either specifiy path or model loaded from h5 via keras.models.load_model")
    if stride == None:
        stride = [[4.0, 4.0, 4.0] for _ in range(len(geometry_filenames))]
    if box_size == None:
        box_size = [16.0 for _ in range(len(geometry_filenames))]
    
    if format_ == None:
        if mode == 'global':
            format_ = 'aims'
        elif mode == 'local':
            format_ = 'xyz'
            
    if not model == None:
        try:
            model_file_ending = model.split('.')[1]
            if not model_file_ending == '.h5':
                raise NotImplementedError("Model path must link to h5 file.")
        except:
            raise ValueError("Model must be a path to a h5 file or None. In the latter case, a pretrained model is loaded.")
    
    if mode == 'global':
        predictions, uncertainty = global_(geometry_filenames, descriptor=descriptor,
                                           model=model, format_=format_,
                                           descriptors=descriptors, save_descriptors=save_descriptors,
                                           save_path_descriptors=save_path_descriptors, **kwargs)
    elif mode == 'local':
        predictions, uncertainty = local(geometry_filenames, stride, box_size, configs,
                                         descriptor=descriptor, model=model, format_=format_,
                                         desc_filename=descriptors, nb_jobs=nb_jobs, **kwargs)
    else:
        raise ValueError("Argument 'mode' must either be 'local' or 'global'.")
    return predictions, uncertainty
