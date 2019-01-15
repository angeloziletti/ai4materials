#!/usr/bin/python
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
from future.utils import iteritems

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016-2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/08/17"

import ase.io
from ase.db import connect
import collections
import errno
import json
import logging
import math
from ai4materials.utils.utils_crystals import get_spacegroup_analyzer
import numpy as np
import os
import pandas as pd
from PIL import Image
import PIL.ImageOps
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import six.moves.cPickle as pickle
import tarfile
logger = logging.getLogger('ai4materials')


def check_structure_list(structure_list):
    """Check if the descriptor name is the same for all structures in the list"""
    desc_names = [item.info['descriptor']['descriptor_name'] for item in structure_list]

    # this would short-circuit (i.e. return false as soon as the predicate fails for an item instead of going on)
    # from https://stackoverflow.com/questions/4752294/all-list-values-same
    if not all(desc_names[0] == desc_name for desc_name in desc_names):
        raise AssertionError('Descriptor names in the list of structures are different. \n'
                             'List of descriptor names: {0}'.format(desc_names))

    return True


def get_metadata_value(structure_list, desc_metadata):
    """Retrieve a given metadata from a list of structures from the descriptor key"""
    desc_metadata_list = []

    for structure in structure_list:
        desc_metadata_value = structure.info['descriptor'][str(desc_metadata)]
        desc_metadata_list.append(desc_metadata_value)

    return desc_metadata_list


def generate_facets_input(structure_list, desc_metadata, target_list, configs,
                          sprite_atlas_filename='desc_atlas', df_filename='df_facets',
                          make_sprite_atlas=True, normalize=False, **kwargs):
    """Generate facets input.

    Parameters:

    structure_list: list of ``ase.Atoms`` objects
        List of atomic structures.

    desc_metadata: string
        Name of the descriptor metadata to be retrieved from the .
        A list of metadata for each descriptor can be found in `descriptors.nomadmetainfo.json`

    target_list: list of dict
        List of dictionaries as returned by `nomad-ml.wrappers.load_descriptor`. \n
        Each element of this list is a dictionary with only one key (data), \n
        which has as value a list of dicts. \n
        For example: \n
        {u’data’: [{u’spacegroup_symbol_symprec_0.001’: 194, u’chemical_formula’: u’Ac258’}]}. \n
        More keywords are possible.

    configs: dict
        Dictionary containing configuration information such as folders for input and output \n
        (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.\n
        See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    sprite_atlas_filename: string, optional (default = `desc_atlas`)
        Name (without extension) for the generated texture atlas containing the descriptor images.
        The .png extension will be automatically appended to the filename.

    df_filename: string, optional (default = `df_facets`)
        Name (without extension) for the generated Pandas dataframe containing information
        on the list of structures.
        The .csv extension will be automatically appended to the filename.

    make_sprite_atlas: bool, optional (default = `True`)
        If `True`, a texture atlas containing the descriptor images is generated.
        If `False`, only the `df_filename.csv` containing the Pandas dataframe is generated.

    normalize: bool, optional (default = `False`)
        If `True`, each descriptor image is normalized (separately), i.e. all descriptor images
        will have a maximum of 1 and a minimum of 0.

    Return:

    string, string
        Absolute path to the Pandas dataframe file and the texture atlas file (if created, otherwise returns `None`).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    check_structure_list(structure_list)
    logger.info("Descriptor: {}".format(structure_list[0].info['descriptor']['descriptor_name']))
    logger.debug("Available descriptor metadata: {}".format(structure_list[0].info['descriptor'].keys()))
    logger.info("Loading descriptor metadata: {}".format(desc_metadata))

    images_list = get_metadata_value(structure_list, desc_metadata)
    images = np.asarray(images_list)

    if len(images.shape) == 3:
        images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))

    logger.info('Images ({0}) matrix shape: {1}'.format(desc_metadata, images.shape))

    if normalize:
        for idx in range(images.shape[0]):
            images[idx, :, :, :] = (images[idx, :, :, :] - np.amin(images[idx, :, :, :])) / (
                np.amax(images[idx, :, :, :]) - np.amin(images[idx, :, :, :]))

    if make_sprite_atlas:
        sprite_atlas_filename_path = create_sprite_atlas(images=images, main_folder=configs['io']['main_folder'],
                                                         sprite_atlas_filename=sprite_atlas_filename)
    else:
        logger.info("Not creating sprite atlas.")
        sprite_atlas_filename_path = None

    target_list_dict = [item['data'][0] for item in target_list]

    df = pd.DataFrame(target_list_dict)

    # check which columns are dictionaries
    df = expand_dict_labels(df)

    # add lists inside dictionary to DataFrame as new columns
    df = df.assign(**kwargs)

    df_filepath = os.path.abspath(os.path.normpath(os.path.join(configs['io']['main_folder'], df_filename + '.csv')))
    df.to_csv(df_filepath, index=False)

    return df_filepath, sprite_atlas_filename_path


def analize_dataset(df_filepath):
    """Analize the csv generated."""

    logger.debug('Reading file: {0}'.format(df_filepath))

    df = pd.read_csv(df_filepath, sep=',')

    ground_truth = ["spacegroup_number_0.001_1.0", "spacegroup_number_0.01_1.0", "spacegroup_number_0.1_5.0"]

    predictions = ["spacegroup_number_actual_0.001_1.0", "spacegroup_number_actual_0.01_1.0",
                   "spacegroup_number_actual_0.1_5.0"]

    for idx, item in enumerate(predictions):
        y_true = df[ground_truth[idx]].values
        y_pred = df[predictions[idx]].values

        logger.info("Ground truth: '{0}' predictions: '{1}'".format(ground_truth[idx], predictions[idx]))
        logger.info("Accuracy score: {0}%".format(accuracy_score(y_true, y_pred, normalize=True) * 100.0))

        logger.info(df.groupby([ground_truth[idx], predictions[idx]]).size())

        # to print long strings (default=50)
        pd.options.display.max_colwidth = 200

        logger.info("Files from which the parental structure was correctly identified.")
        logger.info(df.loc[df[ground_truth[idx]] == df[predictions[idx]]]['main_json_file_name'])

    return df_filepath


def expand_dict_labels(df):
    """Given a Pandas Dataframe, it expands the columns that are dictionaries. """

    # check which columns are dictionaries
    features_type = df.applymap(type).eq(dict).all().to_dict()

    # expand the features that are dictionaries on multiple columns, one for every key
    for key in features_type:
        if features_type[key]:
            # replace parenthesis and spaces because they cause trouble to Facets
            new_cols_from_dict = [key + "_" + item for item in df[key].apply(pd.Series).columns.tolist()]
            new_cols_from_dict = [item.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_") for item in
                                  new_cols_from_dict]
            df[new_cols_from_dict] = df[key].apply(pd.Series)
            df.drop(key, axis=1, inplace=True)

    return df


def extract_labels(target_list, target_name, target_categorical, disc_type='uniform', n_bins=100, new_labels=None):
    """From a target_list obtained from `nomad-ml.wrappers.load_descriptor`, extract the labels.

    Assumes that all the classes are contained in the training set.
    The test set can have less classes.

    Parameters:

    target_list: list of dict
        List of dictionaries as returned by the `nomad-ml.wrappers.load_descriptor`.
        Each element of this list is a dictionary with only one key (`data`), which has as value a list of dicts.
        For example: \n
        {u'data': [{u'spacegroup_symbol_symprec_0.001': 194, u'chemical_formula': u'Ac258'}]}.\n
        More keywords are possible.

    target_name: str
        Name of the target to extract from the target list. `target_name` needs to be a key in the dict contained in
        `target_list`. For the example above, valid target names are 'spacegroup_symbol_symprec_0.001' and
        'chemical_formula'.

    target_categorical: bool
        If `True`, the target to extract is assumed to be categorical, i.e. for classification.
        If `False`, the target to extract is assumed to be continuous, i.e. for regression.
        If `True`, the labels are discretized according to `disc_type`.

    disc_type: { 'uniform', 'quantiles'}
        Type of discretization used if target is categorical. In both case, `n_bins` are used.

    n_bins: int, optional (default=100)
        Number of bins used in the discretization.

    new_labels: dict, optional (default = `None`)
        It allows to substitute the label names that are in `target_list`.
        For example: \n
        new_labels = {"hcp": ["194"], "fcc": ["225"], "diam": ["227"], "bcc": ["229"]} \n
        will substitute each occurrence of "194" with "hcp" in the label list which is extracted.

    Returns:

    ``sklearn.preprocessing.LabelEncoder``, ndarray, ndarray
        Returns ``sklearn.preprocessing.LabelEncoder`` object, ndarray with the numerical labels, ndarray with the
        text labels.
        The ndarray with the numerical labels is obtained via: \n
            label_encoder = preprocessing.LabelEncoder() \n
            label_encoder.fit(text_labels) \n
            numerical_labels = label_encoder.transform(text_labels)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    target_list_dict = []
    for idx, item in enumerate(target_list):
        # the 0 refers to the first frame
        target_list_dict.append(item["data"][0])

    df = pd.DataFrame(target_list_dict)

    # expand only the columns that are dictionaries.
    df = expand_dict_labels(df)

    if target_name in df.columns.values:
        target = df[target_name]
    else:
        raise ValueError("Target name not present. Possible values are {}.".format(df.columns.tolist()))

    if target_categorical:
        text_labels = np.asarray(target)
    else:
        logger.info('Converting numerical target to categorical.')
        max_value = np.amax(target)
        min_value = np.amin(target)
        logger.info("Target range: [{0}, {1}]".format(min_value, max_value))
        logger.debug("Number of bins: {0}".format(n_bins))
        logger.debug("Discretization type: {0}".format(disc_type))

        if disc_type == 'uniform':
            bins = np.linspace(min_value, max_value, n_bins)
            dx = (abs(min_value - max_value)) / n_bins
            logger.debug("Discretization bin size: {0}".format(dx))
        elif disc_type == 'quantiles':
            bins = list((pd.qcut(target, n_bins, labels=False, retbins=True))[1])
            # Reproduce `histogram` binning by manually shifting the
            # rightmost bin edge by an epsilon value:
            # https://github.com/numpy/numpy/issues/4217
            # not needed for uniform because there the bins are 10, not 11 like here
            bins[-1] += 10E-8
        else:
            raise ValueError("Please specify a valid discretization type (disc_type)."
                             "Possible values are 'uniform' or 'quantiles'.")

        text_labels = np.digitize(target, bins, right=False)

        # calculate just to show (note: bins are re-defined here)
        freq, bins = np.histogram(target, bins=bins)

        logger.debug("Class distribution:")

        for idx in range(len(bins) - 1):
            logger.debug("{0}<= x<{1}  count:{2}".format(bins[idx], bins[idx + 1], freq[idx]))

    # redefine labels
    if new_labels is not None:
        text_labels = [str(item) for item in text_labels]
        for key in new_labels.keys():
            text_labels = [key if item in new_labels[key] else item for item in text_labels]
        text_labels = np.asarray(text_labels)

    # from class_labels to class number
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(text_labels)
    numerical_labels = label_encoder.transform(text_labels)

    logger.debug("Number of unique classes in the dataset: {0}".format(len(label_encoder.classes_)))
    logger.debug("Actual class labels: \n {0}".format(list(label_encoder.classes_)))

    return label_encoder, numerical_labels, text_labels


def extract_img_list(filename, desc_folder=None, tmp_folder=None):
    """ Return the list of images from ."""

    logger.debug('Loading file {0}'.format(filename))

    files_by_category = read_archive_desc(filename, summary_filename="summary.json")
    desc_folder = desc_folder + tmp_folder

    img_list = []
    for file_ in files_by_category['2d_diffraction_images_ks']:
        img = os.path.abspath(os.path.normpath(os.path.join(desc_folder, file_)))
        img_list.append(img)

    return img_list


def extract_ase_db_files(filename, tmp_folder=None, clean_tmp=False):
    """ Read target files from descriptor folder.
    Change to put the extractall inside."""

    logger.debug('Loading file {0}'.format(filename))

    if clean_tmp:
        clean_folder(tmp_folder)

    archive = tarfile.open(filename, 'r')
    archive.extractall(tmp_folder)

    data_list = []
    files_by_category = read_archive_desc(filename, summary_filename="summary.json")

    for item in files_by_category['ase_db_files']:
        in_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, item)))
        with open(in_file) as json_file:
            try:
                data = json.load(json_file)
                data_list.append(data)
            except Exception as e:
                logger.error("Could not read content from JSON file {0}".format(in_file))
                logger.error("Error: {0}".format(e))
            finally:
                json_file.close()

    archive.close()

    return data_list


def extract_files(filename, file_type='target_files', tmp_folder=None, clean_tmp=False):
    """ Read files from descriptor folder.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    logger.debug('Loading file {0}'.format(filename))

    if clean_tmp:
        clean_folder(tmp_folder)

    archive = tarfile.open(filename, 'r')
    archive.extractall(tmp_folder)

    data_list = []
    files_by_category = read_archive_desc(filename)

    if file_type == 'target_files':
        for item in files_by_category['target_files']:
            in_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, item)))
            with open(in_file) as json_file:
                try:
                    target = json.load(json_file)
                    data_list.append(target)
                except Exception as e:
                    logger.error("Could not read content from JSON file {0}".format(in_file))
                    logger.error("Error: {0}".format(e))
                finally:
                    json_file.close()

    elif file_type == 'ase_db_files':
        # load the ASE structure, then the associated info dictionary
        for idx, item in enumerate(files_by_category['ase_db_files']):
            ase_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, item)))
            structure = ase.io.read(ase_file, format='json')
            info_filename = files_by_category['ase_info_files'][idx]
            structure_info_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, info_filename)))

            with open(structure_info_file, 'rb') as input_file:
                try:
                    info_dict = pickle.load(input_file)
                    structure.info = info_dict
                    data_list.append(structure)
                except Exception as e:
                    logger.error("Could not read content cPickle from file {0}".format(input_file))
                    logger.error("Error: {0}".format(e))
                finally:
                    input_file.close()
    else:
        raise TypeError("File_type {} not recognized.".format(file_type))
    archive.close()

    return data_list


def create_sprite_atlas(images, main_folder, sprite_atlas_filename='sprite_atlas', max_imgs_row=None):
    """Create a sprite atlas for numpy array of images(nb_images, px, py, nb_channels)
    Based on: https://minzkraut.com/2016/11/23/making-a-simple-spritesheet-generator-in-python/

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    sprite_atlas_filename_path = os.path.abspath(os.path.normpath(os.path.join(main_folder, sprite_atlas_filename)))

    if max_imgs_row is None:
        max_imgs_row = int(math.sqrt(images.shape[0]))

    img_list = []
    for idx in range(images.shape[0]):
        img = images[idx, :, :]
        img_list.append(img)

    nb_imgs, tile_height, tile_width, nb_channels = images.shape

    if nb_imgs > max_imgs_row:
        spritesheet_width = tile_width * max_imgs_row
        required_rows = math.ceil(nb_imgs / max_imgs_row)
        imgs_last_row = nb_imgs % max_imgs_row
        if imgs_last_row != 0:
            required_rows = required_rows + 1
        spritesheet_height = tile_height * required_rows
    else:
        spritesheet_width = tile_width * nb_imgs
        spritesheet_height = tile_height

    if nb_channels == 1:
        mode = 'L'
    elif nb_channels == 3:
        mode = 'RGBA'
    else:
        raise ValueError('Unexpected number of channels ({}).'.format(nb_channels))

    spritesheet = Image.new(mode, [int(spritesheet_width), int(spritesheet_height)])

    for idx, current_img in enumerate(img_list):
        top = tile_height * math.floor(idx / max_imgs_row)
        left = tile_width * (idx % max_imgs_row)
        bottom = top + tile_height
        right = left + tile_width

        box = (left, top, right, bottom)
        box = [int(i) for i in box]

        rgb_array = np.zeros((tile_height, tile_width, nb_channels), 'uint8')

        current_img = list(current_img.reshape(-1, current_img.shape[0], current_img.shape[1]))

        for ix_ch in range(len(current_img)):
            rgb_array[..., ix_ch] = current_img[ix_ch] * 255

        current_frame_img = PIL.Image.fromarray(rgb_array)
        spritesheet.paste(current_frame_img, box)

    sprite_atlas_filename_path = sprite_atlas_filename_path + ".png"
    spritesheet.save(sprite_atlas_filename_path, "PNG")
    logger.info("Image creation completed.")
    logger.info("Saving at {0}".format(sprite_atlas_filename_path))

    return sprite_atlas_filename_path


def clean_folder(tmp_folder, endings_to_delete=(".png", ".npy", "_target.json",
                                                "_aims.in", "_ase_atoms_info.pkl", "_ase_atoms.json",
                                                "_coord.in",
                                                "summary.json", "_atomic_features.csv")):
    """ Clean tmp folder from files that do not end in allowed_endings.

    Parameters:

    tmp_folder: string
        Absolute path to the folder from which we want to erase the files that ends
        with the endings in `endings_to_delete`

    endings_to_delete: tuple of strings, optional (default=(".png", ".npy", "_target.json", "_aims.in"))
        Delete from `tmp_folder` all the files that which filenames end with the mentioned endings.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    list_of_files = os.listdir(tmp_folder)

    flt_files = filter(lambda x: x.endswith(tuple(endings_to_delete)), list_of_files)

    logger.debug("Cleaning {0} folder.".format(tmp_folder))
    for flt_file in flt_files:
        file_path = os.path.join(tmp_folder, flt_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(e)


def read_archive_desc(filename, summary_filename="summary.json"):
    """ Read descriptor file archive and return a dictionary.

    Parameters:

    filename: string
        Absolute path the the descriptor file to be read.

    summary_filename: string, optional (default=`summary.json`)
        Name of the summary file that contains a recap of the content of the
        descriptor archive to be read.

    Returns:

    dict or None
        A dictionary with (key=filetype, value=list_of_files_for_that_type) if the files can be
        successfully loaded, ``None`` otherwise. See :py:mod:`ai4materials.utils.utils_data_retrieval.write_summary_file`
        for the possible file types and corresponding allowed endings.

    .. seealso:: For possible file types see
        :py:mod:`ai4materials.utils.utils_data_retrieval.write_summary_file`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    logger.debug('Loading file {0}'.format(filename))
    archive = tarfile.open(filename, 'r')

    files_by_category = None
    # extract the Archive in tmp folder
    for member in archive.getmembers()[:]:
        member.name = os.path.basename(member.name)
        if os.path.basename(member.name) == summary_filename:
            files_by_category = json.load(archive.extractfile(member).decode('utf-8'))['data'][0]

    return files_by_category


def write_summary_file(descriptor, desc_file_list, tmp_folder, allowed_endings_user=None, desc_file_master=None,
                       summary_filename="summary.json", clean_tmp=False):
    """Write a summary of the files present in the descriptor archive list.

    If more than one descriptor archive is passed, the files present will be merged
    and a summary files containing all files present in all descriptor archives
    will be written. A typical example is when descriptor archives created by different
    processes in a parallel computation are merged by the master in a single
    file containing all the results.

    Parameters:

    descriptor: `ai4materials.descriptors.base_descriptor.Descriptor` object
        Descriptor obejct for which the summary file is written.

    desc_file_list: string or list of strings
        Desc archive or list of descriptor archives for which the summary file is written.

    tmp_folder: string
        Path to the tmp folder.

    allowed_endings_user: dict
        A dictionary containing as key the filetype and as value the allowed endings.
        For example: {'target_files': '_target.json'}
        These allowed endings will be added to the possible filetypes specified in the
        descriptor metadatas and in this function.

    desc_file_master: string
        Full path to the archive where the summary file will be written.

    summary_filename: string, optional (default=`summary.json`)
        Name of the summary file that contains a recap of the content of the
        descriptor file to be read.

    clean_tmp: bool, optional (default=`False`)
        If `True`, clear the `tmp_folder` calling :py:mod:`ai4materials.utils.utils_data_retrieval.clean_folder`
        with argument `tmp_folder`.

    Returns:

    string
        Return the absolute path to the descriptor archive (`desc_file_master`) where the summary file is written.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    allowed_endings = {'target_files': '_target.json', 'ase_db_files': '_ase_atoms.json',
                       'ase_info_files': '_ase_atoms_info.pkl'}
    allowed_endings_desc = descriptor.desc_metadata['file_ending'].to_dict()
    allowed_endings.update(allowed_endings_desc)

    # allowed_endings_user overwrite the defaults and the desc metadata
    if allowed_endings_user is not None:
        allowed_endings.update(allowed_endings_user)

    if not isinstance(desc_file_list, list):
        desc_file_master = desc_file_list
        desc_file_list = [desc_file_list]

    logger.debug('Descriptor file_list: {0}'.format(desc_file_list))
    logger.debug('Descriptor file master: {0}'.format(desc_file_master))

    if clean_tmp:
        clean_folder(tmp_folder)

    archive_list = []
    for desc_file in desc_file_list:
        archive = tarfile.open(desc_file, 'r')
        archive_list.append(archive)

    members = []
    for archive in archive_list:
        for member in archive.getmembers()[:]:
            member.name = os.path.basename(member.name)
            archive.extract(member, tmp_folder)
            members.append(member.name)

    filelist_dict = {}
    for archive in archive_list:
        for member in archive.getmembers()[:]:
            if not member.isfile():
                continue
            # make a dictionary {"filename", "filetype"}
            for (key, value) in iteritems(allowed_endings):
                if member.name.endswith(value):
                    filelist_dict[member.name] = key

        archive.close()

    files_by_cat = collections.defaultdict(list)
    for key, value in sorted(iteritems(filelist_dict)):
        files_by_cat[value].append(key)

    # re-open the Archive to add the summary.json file
    logger.debug('Loading file {0}:'.format(desc_file_master))
    new_archive = tarfile.open(desc_file_master, 'w:gz')

    member_filename_paths = []
    for member in members:
        member_filename_path = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, member)))
        member_filename_paths.append(member_filename_path)

    summary_filename_path = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, summary_filename)))
    member_filename_paths.append(summary_filename_path)

    # get unique list on member_filename_paths to avoid to add multiple times
    # the same file (e.g. summary.json will not be added twice if already present)
    member_filename_paths = sorted(list(set(member_filename_paths)))

    with open(summary_filename_path, "w") as f:
        f.write("""
    {
          "data":[""")
        json.dump(files_by_cat, f, indent=2)
        f.write("""
    ] }""")
        f.flush()

    # add arcname to avoid to include the whole folder path in the member.name
    for member_filename_path in member_filename_paths:
        new_archive.add(member_filename_path, arcname=os.path.basename(member_filename_path))

    logger.debug('Summary file written in {0}:'.format(desc_file_master))

    new_archive.close()

    return desc_file_master


def write_ase_db(ase_atoms_list, main_folder, db_name='my_ase_db', db_type='db',
                 overwrite=True, folder_name='db_ase'):
    """From a list of ASE atom objects, write an ASE database to file.

    It uses ASE database functionality to write to file a database given the structures in ase_atoms_list.
    For more information regarding ASE databases: https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db.core

    db_type: str, optional (default=‘db’)
        One of ‘json’, ‘db’, ‘postgresql’, (JSON, SQLite, PostgreSQL).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    ase_db_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, folder_name)))

    if not os.path.exists(ase_db_folder):
        os.makedirs(ase_db_folder)

    ase_db_filename = os.path.abspath(os.path.normpath(os.path.join(ase_db_folder,
                                                                    '{0}.{1}'.format(db_name, str(db_type)))))

    if overwrite:
        silent_remove(ase_db_filename)

    db = connect(ase_db_filename, type=db_type)

    for idx, atoms in enumerate(ase_atoms_list):
        if idx % (int(len(ase_atoms_list) / 10) + 1) == 0:
            logger.info("Writing: file {0}/{1} "
                        "to ASE database".format(idx + 1, len(ase_atoms_list)))
        # write structure to ASE database
        # other info https://wiki.fysik.dtu.dk/ase/ase/db/db.html?highlight=db#ase-db
        db.write(atoms, data=atoms.info)

    logger.info("ASE database written in file: {}".format(ase_db_filename))

    return ase_db_filename


def read_ase_db(db_path):
    """From the path to an ASE database file, return a list of ASE atom object contained in it.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    db = connect(db_path)

    logger.debug("Database length: {}".format(len(db)))

    ase_list = []
    for idx_db in range(len(db)):
        atoms = db.get_atoms(selection=idx_db + 1, add_additional_information=True)
        # put info from atoms.info['data'] back at their original place (atoms.info)
        # this is because when an ASE atoms object is saved into the SQLite database,
        # ASE does not save automatically atoms.info but instead to
        # atoms.info are saved in atoms.info['data']
        if 'data' in atoms.info.keys():
            atoms.info = atoms.info['data']

        ase_list.append(atoms)

    return ase_list


def write_ase_db_file(structure, configs, tar=None, filename_suffix='_ase_atoms.json',
                      filename_suffix_info='_ase_atoms_info.pkl', op_nb=0):
    desc_folder = configs['io']['desc_folder']

    ase_db_filename = os.path.abspath(os.path.normpath(
        os.path.join(desc_folder, structure.info['label'] + '_' + 'op' + str(op_nb) + filename_suffix)))

    ase_info_filename = os.path.abspath(os.path.normpath(
        os.path.join(desc_folder, structure.info['label'] + '_' + 'op' + str(op_nb) + filename_suffix_info)))

    structure.write(ase_db_filename, format='json')

    with open(ase_info_filename, "w") as f_out:
        f_out.write(pickle.dumps(structure.info))

    structure.info['ase_db_filename'] = ase_db_filename
    structure.info['ase_info_filename'] = ase_info_filename

    if tar is not None:
        tar.add(structure.info['ase_db_filename'])
        tar.add(structure.info['ase_info_filename'])


def group_filter_files(configs, ending, get_key):
    desc_folder = configs['io']['desc_folder']

    filters_keys = dict()
    for root, dirs, files in os.walk(desc_folder, topdown=True):
        for file_ in files:
            if file_.endswith(ending):
                filters_key = get_key(file_)

                if filters_key in filters_keys:
                    # append the new filename to existing array at this slot
                    filters_keys[filters_key].append(os.path.join(root, file_))
                else:
                    # create a new array in this slot
                    filters_keys[filters_key] = [os.path.join(root, file_)]

    return filters_keys


def get_paths_from_filter_dict(filter_dict, accepted_keys=None, verbose=True):
    if accepted_keys is None:
        accepted_keys = filter_dict.keys()

    filters_jsons = dict()
    for key in accepted_keys:
        for filter_file in filter_dict[key]:
            with open(filter_file) as json_file:
                data = json.load(json_file)
                if key in filters_jsons:
                    # append the new filename to existing array at this slot
                    filters_jsons[key].extend(data['data'][0]['filtered_json_list'])
                else:
                    # create a new array in this slot
                    if isinstance(data['data'][0]['filtered_json_list'], list):
                        filters_jsons[key] = data['data'][0]['filtered_json_list']
                    else:
                        filters_jsons[key] = [data['data'][0]['filtered_json_list']]
    if verbose:
        for key, value in filters_jsons.items():
            logger.info('Key: {} \t List length: {}'.format(key, len(value)))

    return filters_jsons


def write_desc_info_file(descriptor, desc_info_file, tar, ase_atoms):
    """Write a file with information regarding the descriptor used in the calculation.

    This is done to ensure that the user can trace back which descriptor was used.

    Parameters:

    descriptor: :py:mod:`ai4materials.descriptors.base_descriptor.Descriptor` object
        Descriptor to calculate.

    desc_info_file: string, optional (default=`None`)
        File where information about the descriptor are written to disk.

    tar: tarfile object
        This is an object obtain as follows: tar = tarfile.open(desc_file, 'w:gz')

    ase_atoms: list of ``ase.Atoms`` objects
            Atomic structures.

    """
    desc_info_file = descriptor.write_desc_info(desc_info_file, ase_atoms)

    tar.add(desc_info_file)

    # remove the desc_info_file since it has been already added to the tar file
    try:
        if os.path.isfile(desc_info_file):
            os.unlink(desc_info_file)
    except Exception as e:
        logger.warning(e)


def write_target_values(structure, configs, op_nb, tar=None, filename_suffix='_target.json',
                        calc_spgroup=True, symprec=[1e-03, 1e-06]):
    """Write target values. One file for each frame.
    The target works only if one frame is considered. Please check.

    Parameters:

    filename_suffix: string, default '_target.json'
        Suffix added after filename

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    desc_folder = configs['io']['desc_folder']

    if not isinstance(symprec, list):
        symprec = [symprec]

    target_filename = os.path.abspath(os.path.normpath(
        os.path.join(desc_folder, structure.info['label'] + '_' + 'op' + str(op_nb) + filename_suffix)))

    # check if target is present
    if 'target' not in structure.info.keys():
        structure.info['target'] = None

    # calculate the actual spacegroup of the structure
    # after the transformations were applied
    spacegroup_analyzer_actual = {}
    if calc_spgroup:
        spacegroup_analyzer_actual = get_spacegroup_analyzer(structure, symprec=symprec)

    out_file = open(target_filename, 'w')

    out_file.write("""
{
      "data":[""")

    result = {"checksum": structure.info['label'], "ase_db_file": structure.info['ase_db_filename'],
              "chemical_formula": structure.get_chemical_formula(), "cell": structure.get_cell().tolist(),
              "filename": target_filename, "target": structure.info['target']}

    # write the spacegroup number obtained before applying the
    # specified transformations
    try:
        for key, value in structure.info['spacegroup_nb'].items():
            spacegroup_symbol_key = 'spacegroup_symbol_' + 'symprec_' + str(key)
            spacegroup_nb_key = 'spacegroup_nb_' + 'symprec_' + str(key)
            result[spacegroup_symbol_key] = value
            result[spacegroup_nb_key] = value

        for key, value in spacegroup_analyzer_actual.items():
            spacegroup_symbol_key = 'spacegroup_symbol_actual_' + 'symprec_' + str(key)
            spacegroup_nb_key = 'spacegroup_nb_actual_' + 'symprec_' + str(key)
            result[spacegroup_symbol_key] = value.get_space_group_symbol()
            result[spacegroup_nb_key] = value.get_space_group_number()
    except BaseException:
        pass

    json.dump(result, out_file, indent=2)

    out_file.write("""
] }""")

    out_file.flush()

    structure.info['target_filename'] = target_filename
    if tar is not None:
        tar.add(structure.info['target_filename'])


def silent_remove(filename):
    """Remove a file that might not exist.

    Taken from https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist.

    """

    try:
        os.remove(filename)
        logger.info("Removing file {}".format(filename))
    except OSError as e:
        # errno.ENOENT = no such file or directory
        if e.errno != errno.ENOENT:
            # re-raise exception if a different error occurred
            raise


def get_json_list(data_folder=None, show_preview=False):
    """ Get the json file list.

    Parameters:

    method: string, {'folder', 'file'}
        Specify how to obtain the json files where the structures are included. \n
        'folder': read from the specified folder all the NOMAD (json) files. \n
        'file': read summary file containing: JSON file, frame number, clustering x_coord, clustering y_coord


        In addition to the file, also the path to the folder where the json files
        (`data_folder`) are stored needs to be given (for now)


    data_folder : string
        Folder where the json files are. .


    Returns:


    list
        List with the absolute paths to the json files.


    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if data_folder is not None:
        json_list = []

        if not isinstance(data_folder, list):
            data_folder = [data_folder]

        for folder in data_folder:
            for root, dirs, files in os.walk(folder, topdown=True):
                for file_ in files:
                    if file_.endswith(".json"):
                        json_list.append(os.path.join(root, file_))

        if show_preview is True:
            json_file = open(json_list[0], 'r')

            # this ensures that even if an exception is raised,
            # the json_file will still be closed properly.
            try:
                data = json.load(json_file)
                logger.debug(json.dumps(data, indent=4, sort_keys=True))

            finally:
                json_file.close()

    else:
        raise Exception("Please specify a valid path to the folder where the data are stored.")

    return json_list
