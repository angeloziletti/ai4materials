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
__date__ = "23/03/18"

from datetime import datetime
import os
import json
import logging
from ai4materials.utils.utils_data_retrieval import extract_labels
from ai4materials.utils.utils_data_retrieval import get_metadata_value
from ai4materials.utils.utils_config import overwrite_configs
import numpy as np
import pyximport
import six.moves.cPickle as pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
import tensorflow as tf

pyximport.install(reload_support=True)
# tf.set_random_seed(0) # tf < 2
tf.random.set_seed(0)
logger = logging.getLogger('ai4materials')


def dense_to_one_hot(labels_dense, label_encoder):
    """Convert class labels from scalars to one-hot vectors.

    Parameters:

    labels_dense: ndarray
        Array that needs to be one-hot encoded.

    label_encoder: `sklearn.preprocessing.LabelEncoder`
        Label encoder object.

    Returns:

    ndarray
        One-hot encoded array of `labels_dense`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    n_classes = len(label_encoder.classes_)
    logger.debug('Unique classes: {0}'.format(n_classes))
    n_labels = labels_dense.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


class DataSet(object):
    """Construct a DataSet.

    Adapted from the TensorFlow tutorial at https://www.tensorflow.org/versions/master/tutorials/index.html
    Should be changed in favor of the Tensorflow dataset."""

    def __init__(self, input_dims, images, labels, dtype=tf.float32, flatten_images=True):
        self._input_dims = input_dims

        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if len(self._input_dims) == 2:
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns*depth]
            if flatten_images:
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])

        elif len(self._input_dims) == 3:
            # Convert shape from [num examples, dim1, dim2, dim3, depth]
            # to [num examples, dim1*dim2*dim3] (assuming depth == 1)
            assert images.shape[4] == 1
            if flatten_images:
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
        else:
            raise Exception("Wrong number of dimensions.")

        if dtype == tf.float32:
            images = images.astype(np.float32)
        else:
            raise Exception('dtype not supported.')

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    # def __getstate__(self):
    #     return self.train.images, self.train.labels, self.val.images, self.val.labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


#    def next_batch(self, batch_size):
#        """Return the next `batch_size` examples from this data set."""
#        start = self._index_in_epoch
#        print start
#        self._index_in_epoch += batch_size
#        if self._index_in_epoch > self._num_examples:
#            # Finished epoch
#            self._epochs_completed += 1
#            # Shuffle the data
#            perm = np.arange(self._num_examples)
#            np.random.shuffle(perm)
#            self._images = self._images[perm]
#            self._labels = self._labels[perm]
#            # Start next epoch
#            start = 0
#            self._index_in_epoch = batch_size
#            assert batch_size <= self._num_examples
#        end = self._index_in_epoch
#        return self._images[start:end], self._labels[start:end]

def make_data_sets(x_train_val, y_train_val, x_test, y_test, split_train_val=False, stratified_splits=True,
                   test_size=None, random_state=None, flatten_images=False, dtype=tf.float32):
    """Given training and test data, make a dataset to be use for analysis.

    Parameters:

    x_train_val: ndarray
        Feature matrix for training or training and validation (depending on the value of `split_train_val`)

    y_train_val: ndarray
        Array containing the labels for training or training and validation (depending on the value of `split_train_val`)

    x_test: ndarray
        Feature matrix for test

    y_test: ndarray
        Array containing the test labels

    split_train_val: bool, optional (default = `False`)
        If `True`, split the `x_train_val` and `y_train_val` in training and validation set.

    stratified_splits: bool, optional (default = `True`)
        If `True`, split the `x_train_val` and `y_train_val` using stratified sampling
        (`sklearn.model_selection.StratifiedShuffleSplit`).

    test_size: float, int, None, optional
        test_size as specified in `sklearn.model_selection.StratifiedShuffleSplit`

    random_state: int, RandomState instance or None, optional (default=None)
        test_size as specified in `sklearn.model_selection.StratifiedShuffleSplit`

    flatten_images: bool, optional (default = `True`)
        If `True`, flatten the `x_train_val` and `x_test` arrays.

    dtype: tf.type (default = `tf.float32`)
        dtype to pass to the dataset class.

    Returns:

    `data_preprocessing.DataSet`
        Return a `data_preprocessing.DataSet` object. This will be change to adopt the standard Tensorflow dataset.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    class DataSets(object):
        pass

    data_sets = DataSets()

    x_train = None
    x_val = None
    y_train = None
    y_val = None

    input_train_val_dims = (x_train_val.shape[1], x_train_val.shape[2])
    input_test_dims = (x_test.shape[1], x_test.shape[2])

    if input_train_val_dims == input_test_dims:
        input_dims = input_train_val_dims
        logger.debug('Input dimension: {}'.format(input_dims))
    else:
        raise Exception('Training/validation and test images have different shapes.\n'
                        'Training/validation images shape: {0}. \n'
                        'Test images shape: {1}. \n'.format(input_train_val_dims, input_test_dims))

    if split_train_val:
        if test_size is None:
            raise ValueError("Cannot split in train and validation if the splitting ratio "
                             "is not provided. Please specify a valid 'test_size'.")

    if split_train_val:
        logger.debug("Splitting in train/validation set")

        if stratified_splits:
            logger.info("Using stratified sampling.")
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            logger.info("Not using stratified sampling.")
            sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, val_index in sss.split(X=x_train_val, y=y_train_val):
            x_train, x_val = x_train_val[train_index], x_train_val[val_index]
            y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        data_sets.train = DataSet(input_dims, x_train, y_train, dtype=dtype, flatten_images=flatten_images)
        data_sets.val = DataSet(input_dims, x_val, y_val, dtype=dtype, flatten_images=flatten_images)

    else:
        data_sets.train = DataSet(input_dims, x_train_val, y_train_val, flatten_images=flatten_images)
        data_sets.val = None

    if (x_test is not None) and (y_test is not None):
        data_sets.test = DataSet(input_dims, x_test, y_test, flatten_images=flatten_images)
    else:
        data_sets.test = None

    return data_sets


def prepare_dataset(structure_list, target_list, desc_metadata, dataset_name, target_name,
                    input_dims, configs, target_categorical=True, dataset_folder=None, desc_folder=None,
                    main_folder=None, tmp_folder=None,
                    disc_type=None, n_bins=100, notes=None, new_labels=None):
    """For a list of `ase.Atoms`, a `target_list`, and a `target_name` creates a dataset and writes it to file.

    Information regarding the dataset are saved in a summary file (ending with "_summary.json"). This includes for
    example creation date, path to the pickles containing the feature matrix (ending with "_x.pkl") and the labels
    (ending with "_y.pkl"), `dataset_name`, `target_name`, `text_labels`, and user-defined notes on the
    dataset.

    The dataset written to file by `ai4materials.preprocessing.prepare_dataset` can be later loaded by
    `ai4materials.preprocessing.load_dataset_from_file`.

    Parameters:

    structure_list: list of `ase.Atoms`
        List of atomic structures.

    target_list: list of dict
        List of dictionaries as returned by `nomad-ml.wrappers.load_descriptor`. \n
        Each element of this list is a dictionary with only one key (data), \n
        which has as value a list of dicts. \n
        For example: \n
        {u’data’: [{u’spacegroup_symbol_symprec_0.001’: 194, u’chemical_formula’: u’Ac258’}]}. \n
        More keywords are possible.

    desc_metadata: str
        Metadata of the descriptor to be extracted from `ase.Atoms.info` dictionary.

    dataset_name: str
        Name to give to the dataset.

    target_name: str
        Name of the target to be extracted from `target_list` and saved in the label pickle.

    target_categorical: bool, optional (default = `True`)
        If `True`, the target to extract is assumed to be categorical, i.e. for classification.\n
        If `False`, the target to extract is assumed to be continuous, i.e. for regression.\n
        If `True`, the labels are discretized according to `disc_type`.

    disc_type: { 'uniform', 'quantiles'}
        Type of discretization used if target is categorical. In both case, `n_bins` are used.
        See also :py:mod:`ai4materials.utils.utils_data_retrieval.extract_labels`.

    n_bins: int, optional (default=100)
        Number of bins used in the discretization.

    configs: dict
        Dictionary containing configuration information such as folders for input and output \n
        (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.\n
        See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    dataset_folder: str, optional (default = `configs['io']['dataset_folder']`)
        Path to the folder where the dataset (two pickles with feature matrix and labels, \n
        plus a summary file in human-readable format) is saved.

    desc_folder: str, optional (default = `configs['io']['desc_folder']`)
        Path to the descriptor folder.

    tmp_folder: str, optional (default = `configs['io']['tmp_folder']`)
        Path to the tmp folder.

    main_folder: str, optional (default = `configs['io']['main_folder']`)
        Path to the main_folder.

    notes: str
        Notes/comments regarding the dataset that will be written in the dataset summary file.

    new_labels: dict, optional (default = `None`)
        It allows to substitute the label names that are in `target_list`. \n
        For example: \n
        new_labels = {"hcp": ["194"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]} \n
        will substitute each occurrence of "194" with "hcp" in the label list which is extracted. \n
        See also :py:mod:`ai4materials.utils.utils_data_retrieval.extract_labels`.

    Returns:

    str, str, str
        Return the path to the feature matrix pickle (numpy.ndarray), the label pickle (numpy.ndarray), \n
        and the human-readable summary file.\n
        This can be read by :py:mod:`ai4materials.preprocessing.load_dataset_from_file`.

    .. seealso:: modules :py:mod:`ai4materials.preprocessing.load_dataset_from_file`, \n
                         :py:mod:`ai4materials.wrappers.load_descriptor`

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    configs = overwrite_configs(configs, dataset_folder=dataset_folder, desc_folder=desc_folder,
                                main_folder=main_folder, tmp_folder=tmp_folder)

    dataset_folder = configs['io']['dataset_folder']

    data_set, nb_classes, label_encoder, numerical_labels, text_labels = merge_labels_data(
        structure_list=structure_list, target_list=target_list, desc_metadata=desc_metadata,
        target_categorical=target_categorical, one_hot=False, flatten_images=False, n_bins=n_bins,
        target_name=target_name, disc_type=disc_type, input_dims=input_dims, split_train_val=False,
        new_labels=new_labels)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    x_name = dataset_name + '_x'
    y_name = dataset_name + '_y'
    summary_name = dataset_name + '_summary'

    path_to_x = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, x_name + '.pkl')))
    path_to_y = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, y_name + '.pkl')))
    path_to_summary = os.path.abspath(os.path.normpath(os.path.join(dataset_folder, summary_name + '.json')))

    # write X and y to file
    with open(path_to_x, 'wb') as output:
        pickle.dump(data_set.images, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Writing x to {0}".format(path_to_x))

    with open(path_to_y, 'wb') as output:
        pickle.dump(data_set.labels, output, pickle.HIGHEST_PROTOCOL)
        logger.info("Writing y to {0}".format(path_to_y))

    now = datetime.now()

    dataset_info = {"creation_date": str(now.isoformat()), "dataset_name": dataset_name, "target_name": target_name,
                    "target_categorical": target_categorical,
                    "disc_type": disc_type, "n_bins": n_bins, "path_to_x": path_to_x, "path_to_y": path_to_y,
                    "path_to_summary": path_to_summary, "nb_classes": nb_classes,
                    "classes": list(label_encoder.classes_), "numerical_labels": numerical_labels.tolist(),
                    "text_labels": text_labels.tolist(), "notes": notes}

    # write summary file with main info about the dataset
    with open(path_to_summary, "w") as f:
        f.write("""
    {
          "data":[""")

        json.dump(dataset_info, f, indent=2)

        f.write("""
    ] }""")

        f.flush()

    logger.info('Summary file written in {0}.'.format(path_to_summary))

    return path_to_x, path_to_y, path_to_summary


def load_dataset_from_file(path_to_x, path_to_y, path_to_summary=None):
    """Read the feature matrix, the labels and the summary of a dataset.

    It reads the dataset written to file by `ai4materials.preprocessing.prepare_dataset`, \n
    and return the feature matrix, the labels and the summary file of a dataset.

    Parameters:

    path_to_x: str
        Path to the pickle file where the feature matrix was saved, \n
        as returned by `ai4materials.preprocessing.prepare_dataset`.

    path_to_y: str
        Path to the pickle file where the feature labels were saved, \n
        as returned by `ai4materials.preprocessing.prepare_dataset`.

    path_to_summary: str, optional (default = `None`)
        Path to the human readable (JSON) dataset summary file \n
        as returned by `ai4materials.preprocessing.prepare_dataset`.

    Returns:

    numpy.ndarray, numpy.ndarray, dict
        Return the feature matrix, the labels, and the human-readable summary file \n
        which were saved with :py:mod:`ai4materials.datapreprocessing.preprocessing.prepare_dataset`.

    .. seealso:: modules :py:mod:`ai4materials.datapreprocessing.preprocessing.prepare_dataset`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    logger.debug("Loading X from {}".format(path_to_x))
    logger.debug("Loading y from {}".format(path_to_y))

    dataset_info = None

    with open(path_to_x, 'rb') as input_x:
        x = pickle.load(input_x)

    with open(path_to_y, 'rb') as input_y:
        y = pickle.load(input_y)

    if path_to_summary is not None:
        with open(path_to_summary, 'rb') as summary_dataset:
            dataset_info = json.load(summary_dataset)

    logger.debug('X-shape: {0}'.format(x.shape))
    logger.debug('y-shape: {0}'.format(y.shape))

    return x, y, dataset_info


def merge_labels_data(structure_list, target_list, desc_metadata, stratified_splits=True, one_hot=True,
                      dtype=tf.float32, flatten_images=False, n_bins=None, target_name=None, target_categorical=None,
                      disc_type=None, input_dims=None, split_train_val=False, test_size=None, random_state=None,
                      new_labels=None):
    """From a list of `ase.Atoms` and target list, merge them in a `data_preprocessing.DataSet` object.

    Parameters:

    structure_list: list of `ase.Atoms`
        List of atomic structures.

    target_list: list of dict
        List of dictionaries as returned by `nomad-ml.wrappers.load_descriptor`. \n
        Each element of this list is a dictionary with only one key (data), \n
        which has as value a list of dicts. \n
        For example: \n
        {u’data’: [{u’spacegroup_symbol_symprec_0.001’: 194, u’chemical_formula’: u’Ac258’}]}. \n
        More keywords are possible.

    desc_metadata: str
        Metadata of the descriptor to be extracted from `ase.Atoms.info` dictionary.

    stratified_splits: bool, optional (default = `True`)
        If `True`, split the `x_train_val` and `y_train_val` using stratified sampling
        (`sklearn.model_selection.StratifiedShuffleSplit`).

    test_size: float, int, None, optional
        test_size as specified in `sklearn.model_selection.StratifiedShuffleSplit`

    random_state: int, RandomState instance or None, optional (default=None)
        test_size as specified in `sklearn.model_selection.StratifiedShuffleSplit`

    flatten_images: bool, optional (default = `True`)
        If `True`, flatten the `x_train_val` and `x_test` arrays.

    dtype: tf.type (default = `tf.float32`)
        dtype to pass to the dataset class.

    target_name: str
        Name of the target to be extracted from `target_list` and saved in the label pickle.

    target_categorical: bool, optional (default = `True`)
        If `True`, the target to extract is assumed to be categorical, i.e. for classification.\n
        If `False`, the target to extract is assumed to be continuous, i.e. for regression.\n
        If `True`, the labels are discretized according to `disc_type`.

    disc_type: { 'uniform', 'quantiles'}
        Type of discretization used if target is categorical. In both case, `n_bins` are used.
        See also :py:mod:`ai4materials.utils.utils_data_retrieval.extract_labels`.

    n_bins: int, optional (default=100)
        Number of bins used in the discretization.

    one_hot: bool, optional (default = `True`)
        Dictionary containing configuration information such as folders for input and output \n
        (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.\n
        See also `ai4materials.dataprocessing.preprocessing.dense_to_one_hot`.

    split_train_val: bool, optional (default = `False`)
        If `True`, split the `x_train_val` and `y_train_val` in training and validation set.

    new_labels: dict, optional (default = `None`)
        It allows to substitute the label names that are in `target_list`. \n
        For example: \n
        new_labels = {"hcp": ["194"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]} \n
        will substitute each occurrence of "194" with "hcp" in the label list which is extracted. \n
        See also :py:mod:`ai4materials.utils.utils_data_retrieval.extract_labels`.

    Returns:

    `data_preprocessing.DataSet`, int, `sklearn.preprocessing.LabelEncoder`, numpy.ndarray, numpy.ndarray
        Return the dataset, number of classes, a label encoder object, \n
        labels as integers (as encoder in the label encoder, \n
        and labels as text (the original labels, not encoded).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    class DataSets(object):
        pass

    data_sets = DataSets()

    x_train = None
    x_val = None
    y_train = None
    y_val = None

    x_list = get_metadata_value(structure_list, desc_metadata)
    x = np.asarray(x_list)

    # extract labels from target_list
    label_encoder, labels, text_labels = extract_labels(target_list=target_list, target_name=target_name,
                                                        target_categorical=target_categorical, disc_type=disc_type,
                                                        n_bins=n_bins, new_labels=new_labels)

    # save labels in numerical labels because labels will change if we have one-hot encoding
    # however, we want to keep track of the label number
    numerical_labels = labels
    nb_classes = len(label_encoder.classes_)

    print_size_np_array(x, 'images')
    print_size_np_array(labels, 'labels')

    class_list, class_pop = np.unique(labels, return_counts=True)

    logger.debug("Class populations: \n {0}".format(class_pop))

    if one_hot:
        labels = dense_to_one_hot(labels, label_encoder=label_encoder)
        logger.debug("Using one-hot encoding. The sample number is {0}*(image matrix samples)".format(nb_classes))

    if split_train_val:

        if stratified_splits:
            logger.info("Using stratified sampling.")
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        else:
            logger.info("Not using stratified sampling.")
            sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, val_index in sss.split(X=x, y=labels):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

        print_size_np_array(x_train, 'x_train')
        print_size_np_array(x_val, 'x_val')
        print_size_np_array(y_train, 'train_labels')
        print_size_np_array(y_val, 'val_labels')

        data_sets.train = DataSet(input_dims, x_train, y_train, dtype=dtype, flatten_images=flatten_images)
        data_sets.val = DataSet(input_dims, x_val, y_val, dtype=dtype, flatten_images=flatten_images)

    else:
        logger.debug("Not splitting in train/validation set")

        print_size_np_array(x, 'images')
        print_size_np_array(labels, 'labels')

        data_sets = DataSet(input_dims, x, labels, dtype=dtype, flatten_images=flatten_images)

    return data_sets, nb_classes, label_encoder, numerical_labels, text_labels


def print_size_np_array(array, array_name):
    """Print shape and total Mb consumed by the elements of the array."""
    logger.debug("Shape of {0} array: {1}".format(array_name, array.shape))
    logger.debug("Size of {0}: {1:.3f} MB".format(array_name, array.nbytes / float(2 ** 20)))


def load_data_from_pickle(path_to_x_train, path_to_x_test, path_to_y_train, path_to_y_test, path_to_x_val=None,
                          path_to_y_val=None):
    """Load data from pickles which contains numpy.ndarray objects."""

    x_val = None
    y_val = None

    with open(path_to_x_train, 'rb') as data_input:
        x_train = pickle.load(data_input)

    with open(path_to_y_train, 'rb') as data_input:
        y_train = pickle.load(data_input)

    if path_to_x_val is not None:
        with open(path_to_x_val, 'rb') as data_input:
            x_val = pickle.load(data_input)

    if path_to_y_val is not None:
        with open(path_to_y_val, 'rb') as data_input:
            y_val = pickle.load(data_input)

    with open(path_to_x_test, 'rb') as data_input:
        x_test = pickle.load(data_input)

    with open(path_to_y_test, 'rb') as data_input:
        y_test = pickle.load(data_input)

    print_size_np_array(x_train, 'x_train')
    print_size_np_array(y_train, 'y_train')

    if path_to_x_val is not None:
        print_size_np_array(x_val, 'x_val')
    else:
        logger.debug('Not loading validation set.')

    if path_to_y_val is not None:
        print_size_np_array(y_val, 'y_val')
    else:
        logger.debug('Not loading Y_validation set.')

    print_size_np_array(x_test, 'x_test')
    print_size_np_array(y_test, 'y_test')

    return x_train, y_train, x_val, y_val, x_test, y_test


def standardize_matrix(matrix, standardize='mean-variance'):
    """Standardize matrix."""

    if standardize is None:
        logger.info('Data not standardized.')
        scaler = preprocessing.StandardScaler(copy=False, with_mean=False, with_std=False).fit(matrix)
    elif standardize == 'mean-variance':
        scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=True).fit(matrix)
        logger.info('Data standardized by removing the mean and scaling to unit variance.')
    elif standardize == 'mean':
        scaler = preprocessing.StandardScaler(copy=False, with_mean=True, with_std=False).fit(matrix)
        logger.info('Data standardized by removing the mean; no scaling to unit variance.')
    elif standardize == 'variance':
        scaler = preprocessing.StandardScaler(copy=False, with_mean=False, with_std=True).fit(matrix)
        logger.info('Data standardized by scaling to unit variance; mean not removed.')
    else:
        raise ValueError("Invalid value for standardize.")

    matrix = scaler.transform(matrix)

    return matrix, scaler
