# coding=utf-8
# Copyright 2016-2018 Angelo Ziletti
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

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "23/09/18"


import keras.backend as K
from keras.models import model_from_json
import logging
import numpy as np
import scipy

logger = logging.getLogger('ai4materials')


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    last_batch_size = len(inputs) % batch_size
    if last_batch_size == 0:
        nb_batch = len(inputs) // batch_size
    else:
        nb_batch = len(inputs) // batch_size + 1

    for start_idx in range(0, nb_batch, 1):
        if start_idx == len(inputs) // batch_size:
            if shuffle:
                excerpt = indices[start_idx:start_idx + last_batch_size]
            else:
                excerpt = slice(start_idx, start_idx + last_batch_size)
            yield inputs[excerpt], targets[excerpt]
        else:
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


def get_decision_boundary(model):
    """ Function to return the x-y coordinates of the decision boundary given a model.

    This assumes the second to last hidden layer is a 2 hidden unit layer with a bias term
    and sigmoid activation on the last layer.

    Code from http://srome.github.io/Visualizing-the-Learning-of-a-Neural-Network-Geometrically/.

    """

    a = model.layers[-1].get_weights()[0][0][0]
    b = model.layers[-1].get_weights()[0][1][0]
    c = model.layers[-1].get_weights()[1][0]
    decision_x = np.linspace(-1, 1, 100)
    decision_y = (scipy.special.logit(.5) - c - a * decision_x) / b
    return decision_x, decision_y


def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    """Obtain the activations for each layer for Keras.

    Works for any kind of model (e.g. recurrent, convolutional, residuals). Not only for images.

    Code from: https://github.com/philipperemy/keras-visualize-activations

    """

    logger.info("Computing activations.")

    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            logger.info(layer_activations.shape)
        else:
            logger.debug(layer_activations)
    return activations


def load_model(model_arch_file, model_weights_file):
    """Load a Keras model."""

    json_file = open(model_arch_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    logger.info('Loading model weights..')
    model.load_weights(model_weights_file)

    return model
