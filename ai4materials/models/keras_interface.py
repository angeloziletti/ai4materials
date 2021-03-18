"""
Idea of this class:
Interface with keras / tensorflow for any version
"""

import numpy as np

from keras import layers
from keras.layers import Dense

import inspect
from inspect import getmembers, isfunction, isclass


import logging
logger = logging.getLogger('ai4materials')

class NN_model():
    
    def __init__(self, params, version):
        """
        params: dict of NN specifications
        version: tensorflow (new) or keras (<=2.4.0)
        
        """

        if len(params) == 0:
            raise ValueError("param dict is empty.")        

        if version == 'keras':
            import keras
        elif version == 'tensorflow':
            import tf.keras as keras
        else:
            raise NotImplementedError("specified keras version not implemented.")
        package_version = keras.__version__
        logger.debug("Current keras version is {}".format(package_version))
        
        try:
            available_layer_classes = dict([_ for _ in getmembers(keras.layers) if isclass(_[1])])
        except:
            raise ValueError("This keras version had some change in the module \
                              structure that does not allow to load all possible layers.")
        
        for layer in params:
            # get layer type
            allowed_layer_types = list(available_layer_classes.keys())
            if not layer in allowed_layer_types:
                raise NotImplementedError("Specified layer not implemented in employed keras version.")
            layer_object = available_layer_classes[layer]
            
            layer_arguments = params[layer]['layer_arguments']
            
            allowed_layer_arguments = layer_object.__init__
            
            for layer_argument in layer_arguments:
                
            
            
            
            # load standard arguments
            
            # if no value specified, get type of not specified argument and set it to some reasonable number (if float, select 0.5, if integer, select 10)
        
            # go through all passed keywords and if no
            # specification is found, just set them to standard
        
    


    def get_default_args(func):
        """
        returns list of non-default values and dictionary of arg_name:default_values for the input function
        From https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
        """
        args, varargs, keywords, defaults = inspect.getargspec(func)
        return args[:-len(defaults)], dict(zip(args[-len(defaults):], defaults))