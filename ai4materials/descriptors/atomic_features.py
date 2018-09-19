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

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016-2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/08/17"

from atomic_data.collections import AtomicCollection
import logging
from mendeleev import element
from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.utils.utils_crystals import convert_energy_substance
from ai4materials.utils.utils_config import read_nomad_metainfo
from ai4materials.utils.utils_config import get_data_filename
import os
import pandas as pd
import string

logger = logging.getLogger('ai4materials')


class AtomicFeatures(Descriptor):
    """Return the atomic features corresponding to a given crystal structure.

    For each atom species present in the crystal structure, a list of atomic features are retrieved
    either from user-defined atomic collection, or the python package Mendeleev (https://pypi.org/project/mendeleev/).
    Example of atomic features are atomic number, electron affinity, ionization potential, orbital radii etc.

    Parameters:

    configs: dict
        Contains configuration information such as folders for input and output
        (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.
        See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    def __init__(self, path_to_collection=None, feature_order_by=None, energy_unit='eV', length_unit='angstrom',
                 materials_class='binaries', configs=None):

        super(AtomicFeatures, self).__init__(configs=configs)

        if feature_order_by is None:
            feature_order_by = 'atomic_mulliken_electronegativity'

        self.feature_order_by = feature_order_by
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.materials_class = materials_class
        self.metadata_info = read_nomad_metainfo()

        if path_to_collection is None:
            path_to_collection = get_data_filename(resource='../tests/ExtendedBinaries_Dimers_Atoms_new.json',
                                                   package='atomic_data')

        self.path_to_collection = path_to_collection

        self.collection = AtomicCollection("binaries", collections=self.path_to_collection)

        logger.info("Reading atomic collection from '{0}'".format(self.path_to_collection))
        if self.feature_order_by is not None:
            logger.info("Ordering atomic features by '{0}' of the elements".format(self.feature_order_by))

    def calculate(self, structure, selected_feature_list=None, **kwargs):

        if selected_feature_list is not None:
            if len(selected_feature_list) < 2:
                raise ValueError("Please select at least two primary features.")
            else:
                self.selected_feature_list = selected_feature_list

        value_list = []

        columns = ['ordered_chemical_symbols']
        features_mendeleev = ['atomic_number']

        chemical_symbols = structure.get_chemical_symbols()

        if self.materials_class == 'binaries':
            # reduce chemical symbols and formula to binary
            chemical_symbols = list(set(chemical_symbols))
            if len(chemical_symbols) == 1:
                chemical_symbols *= 2

        if len(chemical_symbols) != 2:
            raise ValueError("More than 2 different atoms in structure {}. At the moment only structures with one or "
                             "two different chemical species are possible. "
                             "The chemical symbols are {}.".format(structure, chemical_symbols))

        # in a given structure, order by the user-specified atomic_metadata
        p = self.collection.get(self.feature_order_by)
        value_order_by = p.value(chemical_symbols)

        # add lambda because the key that is being used to sort
        # is (val, sym), and not just value.
        # in case of sorting of multiple arrays this is needed
        chemical_symbols = [sym for (val, sym) in
                            sorted(zip(value_order_by, chemical_symbols), key=lambda pair: pair[0])]

        values = [''.join(chemical_symbols)]

        for idx, el_symb in enumerate(chemical_symbols):
            for feature in selected_feature_list:
                # divide features for mendeleev and collection
                if feature in features_mendeleev:
                    elem_mendeleev = element(el_symb)
                    try:
                        value = getattr(elem_mendeleev, feature)
                    except Exception as e:
                        logger.warning("{} not found for element {}.".format(feature, el_symb))
                        logger.warning("{}".format(e))
                        value = float('NaN')

                else:
                    # add features from collection
                    try:
                        p = self.collection.get(feature)
                        value = p.value(el_symb)  # convert to desired units
                        unit = p.value(el_symb, 'units')
                        value = convert_energy_substance(unit, value, energy_unit=self.energy_unit,
                                                         length_unit=self.length_unit)

                    except Exception as e:
                        logger.warning("{} not found for element {}.".format(feature, el_symb))
                        logger.warning("{}".format(e))
                        value = float('NaN')

                values.append(value)
                columns.append(feature + '(' + str(list(string.ascii_uppercase)[idx]) + ')')

        values = tuple(values)
        value_list.append(values)

        atomic_features_table = pd.DataFrame.from_records(value_list, columns=columns)

        # add results in ASE structure info
        descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self),
                               atomic_features_table=atomic_features_table)
        structure.info['descriptor'] = descriptor_data

        return structure

    def write(self, structure, tar, op_id, dict_delta_e=None, path=None, filename_suffix='.json', json_file=None):
        """Given the chemical composition, build the descriptor made of atomic features only."""

        desc_folder = self.configs['io']['desc_folder']

        # make dictionary {primary_feature: value} for each structure
        # dictionary of a dictionary, key: Mat, value: atomic_features
        df = structure.info['descriptor']['atomic_features_table']
        # dict_features = df.set_index('chemical_formula').T.to_dict()

        # filename is the normalized absolute path
        atomic_features_table_filename = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                       structure.info['label'] +
                                                                                       self.desc_metadata.ix[
                                                                                           'atomic_features_table'][
                                                                                           'file_ending'])))

        structure.info['atomic_features_table_filename'] = atomic_features_table_filename
        df.to_csv(structure.info['atomic_features_table_filename'])
        tar.add(structure.info['atomic_features_table_filename'])


def get_table_atomic_features(structures):
    """Starting from atomic structures retrieve the table with atomic features.

    The list of structures must contain the calculated :py:class:`ai4materials.descriptors.atomic_features.AtomicFeatures`.

    Parameters:

    structures: `ase.Atoms` object or list of `ase.Atoms` object
        Atomic structure or list of atomic structure.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if not isinstance(structures, list):
        structures = [structures]

    # create empty dataframe to append to
    df_atomic_features = pd.DataFrame()

    for idx_atoms, ase_atoms in enumerate(structures):
        try:
            df_desc_structure = ase_atoms.info['descriptor']['atomic_features_table']
        except KeyError as err:
            logging.error("{}".format(err))
            logging.error("Key ['descriptor']['atomic_features_table'] not found in structure {}. "
                          "Details on the first structure which caused the error: {}".format(idx_atoms, ase_atoms))

        df_atomic_features = df_atomic_features.append(df_desc_structure, ignore_index=True)

    # check columns with NaNs
    cols_with_nan = df_atomic_features.columns[df_atomic_features.isna().any()].tolist()
    if cols_with_nan:
        logger.info("The following columns contain NaN: {}".format(cols_with_nan))

    return df_atomic_features
