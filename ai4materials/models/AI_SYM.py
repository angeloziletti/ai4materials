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


import logging



def analyze_predictions(geometry_files, predictions, uncertainties, numerical_to_text_label=None, top_n = 3):
    if numerical_to_text_label == None:
        class_info_path = get_data_filename('data/nn_models/AI_SYM_class_info.json')
        with open(class_info_path) as file_name:
            data = json.load(file_name)
        class_labels = data["data"][0]['classes']
        numerical_to_text_label = dict(zip(range(len(class_labels)), class_labels))    
        text_to_numerical_label = dict(zip(class_labels, range(len(class_labels))))
        
        
    for geometry_file, prediction, uncertainty in zip(geometry_files, predictions, uncertainties):
        print('For geometry file {}'.format(geometry_file))
        top_n_indices = np.argsort(prediction.flatten())[-top_n:]
        top_n_indices = np.flip(top_n_indices)
        top_n_probabilities = prediction.flatten()[top_n_indices]
        top_n_labels = [numerical_to_text_label[x_] for x_ in top_n_indices]
        
        for idx in range(top_n):
            #print(idx)
            print('{}. Predicted prototype {} with classification probability {:.4f}'.format(idx+1, top_n_labels[idx], top_n_probabilities[idx]))
        uncertainty_string = ''
        for u_type in uncertainty:
            uncertainty_string += u_type + ' = ' + str(round(uncertainty[u_type], 4)) + ' '
        print('Uncertainty: {}'.format(uncertainty_string))


def preparations(main_folder, p_b_c=False, l_max=6,
                 n_max=9, atom_sigma=0.1, cutoff=4.0,
                 central_weight=0.0,
                 atoms_scaling_cutoffs=[20., 30., 40., 50.], 
                 min_atoms=1, constrain_nn_distances=False, logger=None):
    # read config file
    if logger == None:
        if main_folder is None:
            main_folder = os.getcwd()
        configs = set_configs(main_folder=main_folder)
        logger = setup_logger(configs, level='INFO', display_configs=False)
    else:
        logger = logging.getLogger('ai4materials')

    # set up descriptor
    # Descriptor
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
    model_file = get_data_filename('data/nn_models/AI_SYM_Leitherer_et_al_2021.h5')
    model = load_model(model_file)
    
    return descriptor, model


def global_(geometry_files, main_folder=None, n_iter=1000, logger=None):
    # TODO: change functionality such that at least model prediction is done in parallel (maybe descriptor calculation in parallel as optional since it will create new folders)
    descriptor, model = preparations(main_folder, logger)
    
    input_shape_from_model = model.layers[0].get_input_at(0).get_shape().as_list()[1:]
    target_shape = tuple([-1] + input_shape_from_model)

    predictions = []
    uncertainty = []
    soap_descriptors = []
    structures = []

    for geometry_file in geometry_files:
        # read structure into ase object
        structure = read(geometry_file, ':', format='aims')[0] # TODO more general formatting
                         #format=geometry_file.split('.')[-1])
        print(structure)
        structures.append(structures)
        # calculate descriptor
        soap_desc = descriptor.calculate(structure).info['descriptor']['SOAP_descriptor']
        soap_descriptors.append(soap_desc)
        # calculate predictions
        data = np.reshape(soap_desc, target_shape)
        prediction, uncertainty_ = predict_with_uncertainty(data, model=model, model_type='classification', n_iter=n_iter)
        predictions.append(prediction)
        uncertainty.append(uncertainty_)

    # TODO save predictions, uncertainty, structures, descriptors into ase database file
    return predictions, uncertainty
    
    
def local(geometry_files, stride_size, box_size, n_iter=1000, main_folder=None):
    
    # read config file
    if main_folder == None:
        main_folder = os.getcw()
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)
    
    predictions = []
    uncertainty = []
    
    for geometry_file in geometry_files:
        pass
    
    return predictions, uncertainty
    
def predict(geometry_filenames, mode='global', **kwargs):
    
    if mode == 'global':
        predictions, uncertainty = global_(geometry_filenames, **kwargs)
    elif mode == 'local':
        predictions, uncertainty = local(geometry_filenames, **kwargs)
    else:
        raise ValueError("Argument 'mode' must either be 'local' or 'global'.")
    return predictions, uncertainty
