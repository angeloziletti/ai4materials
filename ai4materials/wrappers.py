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

import os
import json
import tarfile
import pandas as pd
import numpy as np
import copy
import multiprocessing

from ai4materials.utils.utils_mp import dispatch_jobs
from ai4materials.utils.utils_mp import collect_desc_folders
from ai4materials.utils.utils_data_retrieval import clean_folder
from ai4materials.utils.utils_data_retrieval import write_desc_info_file
from ai4materials.utils.utils_config import overwrite_configs
# from ai4materials.utils.utils_config import read_nomad_metainfo
from ai4materials.utils.utils_crystals import modify_crystal
#from ai4materials.models.l1_l0 import combine_features, l1_l0_minimization
#from ai4materials.models.sis import SIS
from ai4materials.utils.utils_data_retrieval import extract_files
from ai4materials.utils.utils_config import get_metadata_info
from ai4materials.utils.utils_data_retrieval import write_ase_db_file
from ai4materials.utils.utils_data_retrieval import write_target_values
from ai4materials.utils.utils_data_retrieval import write_summary_file
from ai4materials.utils.utils_mp import parallel_process
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

logger = logging.getLogger('ai4materials')


def calc_descriptor_in_memory(descriptor, configs, desc_file, ase_atoms_list, tmp_folder=None, desc_folder=None,
                        desc_info_file=None, target_list=None, operations_on_structure=None, nb_jobs=-1, **kwargs):
    """ Calculates the descriptor for a list of atomic structures.

    Starting from a list of ASE structures, calculates for each file the descriptor
    specified by ``descriptor``, and stores the results in the compressed archive
    desc_file in the directory `desc_folder`.

    Parameters:

    descriptor: :py:mod:`ai4materials.descriptors.base_descriptor.Descriptor` object
        Descriptor to calculate.

    configs: dict
        Contains configuration information such as folders for input and output (e.g. desc_folder, tmp_folder),
        logging level, and metadata location. See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    ase_atoms_list: list of ``ase.Atoms`` objects
        Atomic structures.

    desc_file: string
        Name of the compressed archive where the file containing the descriptors are written.

    desc_folder: string, optional (default=`None`)
        Folder where the desc_file is written. If not specified, the desc_folder in read from
        ``configs['io']['desc_folder']``.

    tmp_folder: string, optional (default=`None`)
        Folder where the desc_file is written. If not specified, the desc_folder in read from
        ``configs['io']['tmp_folder']``.

    desc_info_file: string, optional (default=`None`)
        File where information about the descriptor are written to disk.

    target_list: list, optional (default=`None`)
        List of target values. These values are saved to disk when the descriptor is calculated, and they can loaded
        for subsequent analysis.

    operations_on_structure: list of objects
        List of operations to be applied to the atomic structures before calculating the descriptor.

    nb_jobs: int, optional (default=-1)
        Number of processors to use in the calculation of the descriptor.
        If set to -1, all available processors will be used.


    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if desc_info_file is None:
        desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))

    desc_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))

    # make the log file empty (do not erase it because otherwise
    # we have problems with permission on the Docker image)
    outfile_path = os.path.join(tmp_folder, 'output.log')
    open(outfile_path, 'w+')

    # remove control file from a previous run
    old_control_files = [f for f in os.listdir(tmp_folder) if f.endswith('control.json')]
    for old_control_file in old_control_files:
        file_path = os.path.join(desc_folder, old_control_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(e)

    tar = tarfile.open(desc_file, 'w:gz')

    if nb_jobs == -1:
        nb_jobs = min(len(ase_atoms_list), multiprocessing.cpu_count())

    # overwrite configs (priority is given to the folders defined in the function)
    # if desc_folder and tmp_folder are None, then configs are not overwritten
    configs = overwrite_configs(configs=configs, desc_folder=desc_folder, tmp_folder=tmp_folder)

    # define desc_folder and tmp_folder for convenience
    desc_folder = configs['io']['desc_folder']
    tmp_folder = configs['io']['tmp_folder']

    with ProcessPoolExecutor(max_workers=nb_jobs) as executor:
        ase_atoms_list_with_op_nested = executor.map(worker_apply_operations,
                                                     ((ase_atoms, operations_on_structure) for ase_atoms in
                                                      ase_atoms_list))
    ase_atoms_list_with_op = [item for sublist in ase_atoms_list_with_op_nested for item in sublist]

    # check if all elements in the ase list have labels (needed for traceability later)
    label_present = [True if 'label' in ase_atoms.info.keys() else False for ase_atoms in ase_atoms_list_with_op]
    if not np.all(label_present):
        logger.info("Some structures in the list do not have labels. Adding or substituting labels.")
        logger.info("Default labels given by the order in the list (1st structure: label=struct-1)")
        logger.info("To avoid this add a label to each ASE structure using ase_atoms.info['label']='your_label'")

        # substitute and add default labels
        for idx, ase_atoms in enumerate(ase_atoms_list_with_op):
            ase_atoms.info['label'] = str('struct-' + str(idx))

    logger.info('Using {} processors'.format(nb_jobs))

    # load descriptor metadata
    desc_metainfo = get_metadata_info()
    allowed_descriptors = desc_metainfo['descriptors']

    # add target to structures in the list
    if target_list is not None:
        for idx_atoms, ase_atoms in enumerate(ase_atoms_list_with_op):
            ase_atoms.info['target'] = target_list[idx_atoms]

    if descriptor.name in allowed_descriptors:
        logger.info("Calculating descriptor: {0}".format(descriptor.name))

        worker_calc_descriptor = partial(calc_descriptor_one_structure, descriptor=descriptor,
                                         allowed_descriptors=allowed_descriptors, configs=configs, idx_slice=0,
                                         desc_file=desc_file, desc_folder=desc_folder, desc_info_file=desc_info_file,
                                         tmp_folder=tmp_folder, target_list=target_list, **kwargs)

        ase_atoms_results = parallel_process(ase_atoms_list_with_op, worker_calc_descriptor, nb_jobs=nb_jobs)

    else:
        raise ValueError("Please provided a valid descriptor. Valid descriptors are {}".format(allowed_descriptors))

    logger.info("Calculation done.")

    logger.info('Writing descriptor information to file.')

    for idx_atoms, ase_atoms in enumerate(ase_atoms_results):
        descriptor.write(ase_atoms, tar=tar, op_id=0)
        write_ase_db_file(ase_atoms, configs, tar=tar, op_nb=0)

        # we assume that the target value does not change with the application of the operations
        write_target_values(ase_atoms, configs, op_nb=0, tar=tar)

    # write descriptor info to file for future reference
    write_desc_info_file(descriptor, desc_info_file, tar, ase_atoms_results)

    tar.close()

    desc_file_master = write_summary_file(descriptor, desc_file, tmp_folder,
                                          desc_file_master=desc_file + '.tar.gz', clean_tmp=False)

    clean_folder(tmp_folder)
    clean_folder(desc_folder,
                 endings_to_delete=(".png", ".npy", "_target.json", "_aims.in", "_info.pkl", "_coord.in",
                                    "_ase_atoms.json"))

    logger.info('Descriptor file: {}'.format(desc_file_master))

    return desc_file_master


def calc_descriptor_one_structure(ase_atoms, descriptor, **kwargs):
    return descriptor.calculate(ase_atoms, **kwargs)


def _calc_descriptor(ase_atoms_list, descriptor, configs, idx_slice=0, desc_file=None, desc_folder=None,
                     desc_info_file=None, tmp_folder=None, target_list=None, cell_type=None, **kwargs):
    if desc_file is None:
        desc_file = 'descriptor.tar.gz'

    if desc_info_file is None:
        desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))

    if target_list is None:
        target_list = [None] * len(ase_atoms_list)

    desc_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, desc_file)))

    # make the log file empty (do not erase it because otherwise
    # we have problems with permission on the Docker image)
    outfile_path = os.path.join(tmp_folder, 'output.log')
    open(outfile_path, 'w+')

    # remove control file from a previous run
    old_control_files = [f for f in os.listdir(tmp_folder) if f.endswith('control.json')]
    for old_control_file in old_control_files:
        file_path = os.path.join(desc_folder, old_control_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(e)

    tar = tarfile.open(desc_file, 'w:gz')

    # load descriptor metadata
    desc_metainfo = get_metadata_info()
    allowed_descriptors = desc_metainfo['descriptors']

    logger.info("Calculating descriptor: {0}".format(descriptor.name))
    logger.debug("Using {0} cell".format(cell_type))

    ase_atoms_result = []

    for idx_atoms, ase_atoms in enumerate(ase_atoms_list):
        if idx_atoms % (int(len(ase_atoms_list) / 10) + 1) == 0:
            logger.info("Calculating descriptor (process # {0}):  {1}/{2}".format(idx_slice, idx_atoms + 1,
                                                                                  len(ase_atoms_list)))
        # add target list to structure
        ase_atoms.info['target'] = target_list[idx_atoms]

        if descriptor.name in allowed_descriptors:
            structure_result = descriptor.calculate(ase_atoms, **kwargs)
            descriptor.write(ase_atoms, tar=tar, op_id=0)
            write_ase_db_file(ase_atoms, configs, tar=tar, op_nb=0)

            # we assume that the target value does not change with the application of the operations
            write_target_values(ase_atoms, configs, op_nb=0, tar=tar)
            ase_atoms_result.append(structure_result)

        else:
            raise ValueError("Please provided a valid descriptor. Valid descriptors are {}".format(allowed_descriptors))

    logger.debug('Writing descriptor information to file.')

    # write descriptor info to file for future reference
    descriptor.write_desc_info(desc_info_file, ase_atoms_result)

    tar.close()

    # open the Archive and write summary file
    # here we substitute the full path with the basename to be put in the tar archive
    # TO DO: do it before, when the files are added to the tar
    write_summary_file(descriptor, desc_file, tmp_folder)

    logger.info('Descriptor calculation (process #{0}): done.'.format(idx_slice))

    filelist = []

    for file_ in filelist:
        file_path = os.path.join(desc_folder, file_)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(e)

    return desc_file


def load_descriptor(desc_files, configs):
    """Load a descriptor file"""

    if not isinstance(desc_files, list):
        desc_files = [desc_files]

    target_list = []
    structure_list = []
    for idx, desc_file in enumerate(desc_files):
        if idx % (int(len(desc_files) / 10) + 1) == 0:
            logger.info("Extracting file {}/{}: {}".format(idx + 1, len(desc_files), desc_file))
        target = extract_files(desc_file, file_type='target_files', tmp_folder=configs['io']['tmp_folder'])
        structure = extract_files(desc_file, file_type='ase_db_files', tmp_folder=configs['io']['tmp_folder'])
        target_list.extend(target)
        structure_list.extend(structure)

    # This is removed otherwise the files created by the Viewer will be erased
    # clean_folder(configs['io']['tmp_folder'], delete_all=True)

    return target_list, structure_list


def calc_model(method, tmp_folder, results_folder, combine_features_with_ops=True, cols_to_drop=None, df_features=None,
               target=None, energy_unit=None, length_unit=None, allowed_operations=None, derived_features=None,
               max_dim=None, sis_control=None, control_file=None, lookup_file=None):
    """ Calculates model.

    """

    if control_file is None:
        control_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, 'control.json')))

    if max_dim is None:
        max_dim = 3

    if method == 'l1_l0' or method == 'SIS':
        raise NotImplementedError("Not supported currently.")
        # # if there are nan, drop entire row
        # if df.isnull().values.any():
        #     #df.dropna(axis=0, how='any', inplace=True)
        #     #df.reset_index(inplace=True)
        #     logger.info('Dropping samples for which the selected features are not available.')
        if method == 'l1_l0':
            if cols_to_drop is not None:
                df_not_features = df_features[cols_to_drop]
                df_features.drop(cols_to_drop, axis=1, inplace=True)
            else:
                df_not_features = None

        # convert numerical columns in float
        for col in df_features.columns.tolist():
            df_features[str(col)] = df_features[str(col)].astype(float)

        # make dict with metadata name: shortname
        features = df_features.columns.tolist()
        features = [feature.split('(', 1)[0] for feature in features]

        shortname = []
        metadata_info = read_nomad_metainfo()
        for feature in features:
            try:
                shortname.append(metadata_info[str(feature)]['shortname'])
            except KeyError:
                shortname.append(feature)

        features_shortnames = dict(zip(features, shortname))

        if method == 'l1_l0':
            # combine features
            if combine_features_with_ops:
                df_combined = combine_features(df=df_features, energy_unit=energy_unit, length_unit=length_unit,
                                               metadata_info=metadata_info, allowed_operations=allowed_operations,
                                               derived_features=derived_features)
            else:
                df_combined = df_features

            feature_list = df_combined.columns.tolist()

            # replace metadata info name with shortname
            # using the {'metadata_name': 'metadata_shortname'}
            for fullname, shortname in features_shortnames.items():
                feature_list = [item.replace(fullname.lower(), shortname) for item in feature_list]

            # it is a list of panda dataframe:
            # 1st el: 1D, 2nd: 2d, 3rd 3D
            try:
                # feature selection using l1-l0
                df_desc, y_pred, target = l1_l0_minimization(target, df_combined.values, feature_list,
                                                             energy_unit=energy_unit, max_dim=max_dim, print_lasso=True,
                                                             lassonumber=25, lambda_grid_points=100)

            except ValueError as err:
                logger.error("Please select a different set of features and/or operations. ")
                logger.error("Hint: try to remove Z_val, r_sigma, r_pi and/or the x+y / |x+y| operation.")
                logger.error("and/or use [energy]=eV and [length]=angstrom.")
                raise Exception("{}".format(err))

            # write results to file(s)
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)

            for idx_dim in range(max_dim):
                # define paths for file writing
                path_to_csv = os.path.join(results_folder, str(method) + '_dim' + str(idx_dim) + '.csv')
                path_to_csv_viewer = os.path.join(results_folder,
                                                  str(method) + '_dim' + str(idx_dim) + '_for_viewer.csv')

                df_dim = df_desc[idx_dim]
                # add y_pred y_true (i.e. target) to dataframe
                y_pred_true = [('y_pred', y_pred[idx_dim]), ('y_true', target)]
                df_true_pred = pd.DataFrame.from_items(y_pred_true)
                df_result = pd.concat([df_dim, df_true_pred], axis=1, join='inner')

                # extract only the coordinates for the viewer and rename them
                coord_cols = range(idx_dim + 1)
                df_result_viewer = df_dim[df_dim.columns[coord_cols]]
                df_result_viewer.columns = ['coord_' + str(i) for i in coord_cols]
                df_result_viewer = pd.concat([df_result_viewer, df_true_pred], axis=1, join='inner')

                # add columns that were dropped because they are not features
                if cols_to_drop is not None:
                    if df_not_features is not None:
                        df_result = pd.concat([df_result, df_not_features], axis=1, join='inner')
                        df_result_viewer = pd.concat([df_result_viewer, df_not_features], axis=1, join='inner')

                df_result.to_csv(path_to_csv)
                df_result_viewer.to_csv(path_to_csv_viewer)

        if method == 'SIS':
            feature_matrix = df_features.values
            feature_list = df_features.columns.tolist()
            # replace metadata info name with shortname
            # using the {'metadata_name': 'metadata_shorname'}
            for fullname, shortname in features_shortnames.items():
                feature_list = [item.replace(fullname.lower(), shortname) for item in feature_list]

            # get units of each feature from feature list determiend by F_unit.
            # feature_unit_classes is a list of integers (unit classes).
            F_unit = [['IP(A)', 'IP(B)', 'EA(A)', 'EA(B)'], ['E_HOMO(A)', 'E_HOMO(B)', 'E_LUMO(A)', 'E_LUMO(B)'],
                      ['r_s(A)', 'r_s(B)', 'r_p(A)', 'r_p(B)', 'r_d(A)', 'r_d(B)', 'r_sigma(AB)', 'r_pi(AB)'],
                      ['Z(A)', 'Z(B)', 'Z_val(A)', 'Z_val(B)', 'period(A)', 'period(B)'], ['d(AB)', 'd(A)', 'd(B)'],
                      ['E_b(AB)', 'E_b(A)', 'E_b(B)'], ['HL_gap(AB)', 'HL_gap(A)', 'HL_gap(B)']]
            feature_unit_classes = [i_class for f in feature_list for i_class, dimension_group in enumerate(F_unit) if
                                    f in dimension_group]

            # Dictionaries with parameters for initialization and local paths, remote paths, and SIS parameters
            sis = SIS(target, feature_matrix, feature_list, feature_unit_classes=feature_unit_classes,
                      target_unit=energy_unit, control=sis_control,
                      output_log_file='/home/beaker/.beaker/v1/web/tmp/output.log', rm_existing_files=True)

            # start
            sis.start()

            # results
            results = sis.get_results()
            df_desc = [dic['D'] for dic in results]
            y_pred = [dic['P_pred'] for dic in results]

            x_coord = df_desc[1].iloc[:, 0].values
            y_coord = df_desc[1].iloc[:, 1].values

            # write a lookup file for the viewer
            lookup_out = []
            for i in range(df_features.shape[0]):
                lookup_out.append((df_features['json_file_path'][i], df_features['frame_number'][i],  # x_coord
                                   str(x_coord[i]),  # y_coord
                                   str(y_coord[i]),  # target value
                                   df_features['target'][i],  # chemical_formuladiff_
                                   'NaN', df_features['chemical_formula'][i],  # predicted value (2D descriptor)
                                   str(y_pred[1][i])))

            # write log file with general info about the model
            with open(lookup_file, "w") as f:
                f.write("\n".join([" ".join(x) for x in lookup_out]))

        logger.info("Selecting optimal descriptors: done.")

        x_axis_label = df_desc[1].columns.values.tolist()[0]
        y_axis_label = df_desc[1].columns.values.tolist()[1]

        with open(control_file, "w") as f:
            f.write("""
    {
          "model_info":[""")
            model_info = {"x_axis_label": x_axis_label, "y_axis_label": y_axis_label, }
            json.dump(model_info, f, indent=2)
            f.write("""] }""")
            f.flush()


def _apply_operations(ase_atoms, operations_on_structure=None):
    """Apply operations to an ASE atoms class instance."""

    ase_atoms_list_op = []

    if operations_on_structure is not None:
        if not isinstance(operations_on_structure, list):
            operations_on_structure = [operations_on_structure]

        for idx, operation in enumerate(operations_on_structure):
            # use deep copy to create a new instance otherwise we always get
            # only the original structure in the list
            ase_atoms_to_modify = copy.deepcopy(ase_atoms)
            ase_atoms_list_op.append(copy.deepcopy(modify_crystal(ase_atoms_to_modify, operation[0], **operation[1])))

    else:
        ase_atoms_list_op = [ase_atoms]

    return ase_atoms_list_op


def worker_apply_operations(arg):
    ase_atoms, operations_on_structure = arg
    return _apply_operations(ase_atoms, operations_on_structure)


def calc_descriptor(descriptor, configs, desc_file, ase_atoms_list, tmp_folder=None, desc_folder=None,
                    desc_info_file=None, target_list=None, operations_on_structure=None, nb_jobs=-1, **kwargs):
    """ Calculates the descriptor for a list of atomic structures.

    Starting from a list of ASE structures, calculates for each file the descriptor
    specified by ``descriptor``, and stores the results in the compressed archive
    desc_file in the directory `desc_folder`.
    It uses multiprocessing.Pool to parallelize the calculation.

    Parameters:

    descriptor: :py:mod:`ai4materials.descriptors.base_descriptor.Descriptor` object
        Descriptor to calculate.

    configs: dict
        Contains configuration information such as folders for input and output (e.g. desc_folder, tmp_folder),
        logging level, and metadata location. See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    ase_atoms_list: list of ``ase.Atoms`` objects
        Atomic structures.

    desc_file: string
        Name of the compressed archive where the file containing the descriptors are written.

    desc_folder: string, optional (default=`None`)
        Folder where the desc_file is written. If not specified, the desc_folder in read from
        ``configs['io']['desc_folder']``.

    tmp_folder: string, optional (default=`None`)
        Folder where the desc_file is written. If not specified, the desc_folder in read from
        ``configs['io']['tmp_folder']``.

    desc_info_file: string, optional (default=`None`)
        File where information about the descriptor are written to disk.

    target_list: list, optional (default=`None`)
        List of target values. These values are saved to disk when the descriptor is calculated, and they can loaded
        for subsequent analysis.

    operations_on_structure: list of objects
        List of operations to be applied to the atomic structures before calculating the descriptor.

    nb_jobs: int, optional (default=-1)
        Number of processors to use in the calculation of the descriptor.
        If set to -1, all available processors will be used.


    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if nb_jobs == -1:
        nb_jobs = min(len(ase_atoms_list), multiprocessing.cpu_count())

    # overwrite configs (priority is given to the folders defined in the function)
    # if desc_folder and tmp_folder are None, then configs are not overwritten
    configs = overwrite_configs(configs=configs, desc_folder=desc_folder, tmp_folder=tmp_folder)

    # define desc_folder and tmp_folder for convenience
    desc_folder = configs['io']['desc_folder']
    tmp_folder = configs['io']['tmp_folder']

    pool = multiprocessing.Pool(processes=nb_jobs)
    ase_atoms_list_with_op_nested = pool.map(worker_apply_operations,
                                             ((ase_atoms, operations_on_structure) for ase_atoms in ase_atoms_list))
    ase_atoms_list_with_op = [item for sublist in ase_atoms_list_with_op_nested for item in sublist]
    pool.close()
    pool.join()

    # check if all elements in the ase list have labels (needed for traceability later)
    label_present = [True if 'label' in ase_atoms.info.keys() else False for ase_atoms in ase_atoms_list_with_op]
    if not np.all(label_present):
        logger.info("Some structures in the list do not have labels. Adding or substituting labels.")
        logger.info("Default labels given by the order in the list (1st structure: label=struct-1)")
        logger.info("To avoid this add a label to each ASE structure using ase_atoms.info['label']='your_label'")

        # substitute and add default labels
        for idx, ase_atoms in enumerate(ase_atoms_list_with_op):
            ase_atoms.info['label'] = str('struct-' + str(idx))

    def _calc_descriptor_mp(data_slice, desc_file_i, idx_slice):
        _calc_descriptor(ase_atoms_list=data_slice, desc_file=desc_file_i, idx_slice=idx_slice, descriptor=descriptor,
                         configs=configs, logger=logger, tmp_folder=tmp_folder, desc_folder=desc_folder,
                         desc_info_file=desc_info_file, target_list=target_list, **kwargs)

    logger.info('Using {} processors'.format(nb_jobs))
    dispatch_jobs(_calc_descriptor_mp, ase_atoms_list_with_op, nb_jobs=nb_jobs, desc_folder=desc_folder,
                  desc_file=desc_file)

    desc_file_master = collect_desc_folders(descriptor=descriptor, desc_folder=desc_folder, nb_jobs=nb_jobs,
                                            tmp_folder=tmp_folder, desc_file=desc_file, remove=True)

    # the cleaning of the tmp folder does not work if it is put here
    clean_folder(tmp_folder)
    clean_folder(desc_folder, endings_to_delete=(
    ".png", ".npy", "_target.json", "_aims.in", "_info.pkl", "_coord.in", "_ase_atoms.json"))

    logger.info('Descriptor file: {}'.format(desc_file_master))

    return desc_file_master

