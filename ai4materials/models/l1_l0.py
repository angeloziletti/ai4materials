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

import logging
import os
import json
import sys
import pandas as pd
import numpy as np
import random
import math
import itertools
import scipy.stats
from sklearn import linear_model
from math import exp, sqrt
import nomadcore.unit_conversion.unit_conversion as uc
logger = logging.getLogger('ai4materials')


def choose_atomic_features(selected_feature_list=None,
                           atomic_data_file=None, binary_data_file=None):
    """Choose primary features for the extended lasso procedure."""
    df1 = pd.read_csv(atomic_data_file, index_col=False)
    df2 = pd.read_csv(binary_data_file, index_col=False)

    # merge two dataframes on Material
    df = pd.merge(df1, df2, on='Mat')

    # calculate r_sigma and r_pi [Phys. Rev. Lett. 33, 1095(1974)]
    radii_s_p = ['rp(A)', 'rs(A)', 'rp(B)', 'rs(B)']
    df['r_sigma'] = df[radii_s_p].apply(r_sigma, axis=1)
    df['r_pi'] = df[radii_s_p].apply(r_pi, axis=1)

    # calculate Es/sqrt(Zval) and Ep/sqrt(Zval)
    e_val_z = ['Es(A)', 'val(A)']
    df['Es(A)/sqrt(Zval(A))'] = df[e_val_z].apply(e_sqrt_z, axis=1)
    e_val_z = ['Es(B)', 'val(B)']
    df['Es(B)/sqrt(Zval(B))'] = df[e_val_z].apply(e_sqrt_z, axis=1)

    e_val_z = ['Ep(A)', 'val(A)']
    df['Ep(A)/sqrt(Zval(A))'] = df[e_val_z].apply(e_sqrt_z, axis=1)
    e_val_z = ['Ep(B)', 'val(B)']
    df['Ep(B)/sqrt(Zval(B))'] = df[e_val_z].apply(e_sqrt_z, axis=1)

    column_list = df.columns.tolist()
    feature_list = column_list

    if 'Mat' in feature_list:
        feature_list.remove('Mat')
    if 'Edim' in feature_list:
        feature_list.remove('Edim')

    logger.debug("Available features: \n {}".format(feature_list))

    df_selected = df[selected_feature_list]
    df_selected.insert(0, 'Mat', df['Mat'])

    if selected_feature_list:
        logger.info("Primary features selected: \n {}".format(selected_feature_list))
    else:
        logger.error("No selected features.")
        sys.exit(1)

    return df_selected


def classify_rs_zb(structure):
    """Classify if a structure is rocksalt of zincblend from a list of NoMaD structure.
    (one json file). Supports multiple frames (TO DO: check that). Hard-coded.

    rocksalt:
    atom_frac1 0.0 0.0 0.0
    atom_frac2 0.5 0.5 0.5

    zincblende:
    atom_frac1 0.0  0.0  0.0
    atom_frac2 0.25 0.25 0.25

    zincblende --> label=0
    rocksalt  --> label=1
    """
    energy = {}
    chemical_formula = {}
    label = {}

    # gIndexRun=0
    # gIndexDesc=1

    for (gIndexRun, gIndexDesc), atoms in structure.atoms.iteritems():
        if atoms is not None:
            energy[gIndexRun, gIndexDesc] = structure.energy_eV[(gIndexRun, gIndexDesc)]
            # energy=1.0
            chemical_formula[gIndexRun, gIndexDesc] = structure.chemical_formula[(gIndexRun, gIndexDesc)]
            # get labels, works only for RS/ZB dataset
            pos_atom_2 = np.asarray(list(structure.scaled_positions.values())).reshape(2, 3)[1, :]
            if all(i < 0.375 for i in pos_atom_2):
                # label='zincblend'
                label[gIndexRun, gIndexDesc] = 0
            else:
                # label='rocksalt'
                label[gIndexRun, gIndexDesc] = 1

            break

    return chemical_formula, energy, label


def get_energy_diff(chemical_formula_list, energy_list, label_list):
    """ Obtain difference in energy (eV) between rocksalt and zincblend structures of a given binary.

    From a list of chemical formulas, energies and labels returns a dictionary
    with {`material`: `delta_e`} where `delta_e` is the difference between the energy
    with label 1 and energy with label 0, grouped by material.
    Each element of such list corresponds to a json file.
    The `delta_e` is exactly what reported in the PRL 114, 105503(2015).


    .. todo:: Check if it works for multiple frames.


    """
    energy_ = []
    chemical_formula_ = []
    label_ = []

    # energy and chemical formula are lists even if only one frame is present
    for i, energy_i in enumerate(energy_list):
        energy_.append(energy_i.values())

    for i, chemical_formula_i in enumerate(chemical_formula_list):
        chemical_formula_.append(chemical_formula_i.values())

    for i, label_i in enumerate(label_list):
        label_.append(label_i.values())

    # flatten the lists
    energy = list(itertools.chain(*energy_))
    chemical_formula = list(itertools.chain(*chemical_formula_))
    label = list(itertools.chain(*label_))

    df = pd.DataFrame()
    df['Mat'] = chemical_formula
    df['Energy'] = energy
    df['Label'] = label

    # generate summary dataframe with lowest zincblend and rocksalt energy
    # zincblend --> label=0
    # rocksalt  --> label=1
    df_summary = df.sort_values(by='Energy').groupby(['Mat', 'Label'], as_index=False).first()

    groupby_mat = df_summary.groupby('Mat')

    dict_delta_e = {}

    for mat, df in groupby_mat:
        # calculate the delta_e (E_RS - E_ZB)
        energy_label_1 = df.loc[df['Label'] == 1].Energy.values
        energy_label_0 = df.loc[df['Label'] == 0].Energy.values

        # if energy_diff>0 --> rs
        # if energy_diff<0 --> zb
        if (energy_label_0 and energy_label_1):
            # single element numpy array --> convert to scalar
            energy_diff = (energy_label_1 - energy_label_0).item(0)
            # divide by 2 because it is the energy_diff for each atom
            energy_diff = energy_diff / 2.0
        else:
            logger.error(
                "Could not find all the energies needed to calculate required property for material '{0}'".format(mat))
            sys.exit(1)

        dict_delta_e.update({mat: (energy_diff, energy_label_0, energy_label_1)})

    return dict_delta_e


def get_lowest_energy_structures(structure, dict_delta_e):
    """Get lowest energy structure for each material and label type.

    Works only with two possible labels for a given material.

    .. todo:: Check if it works for multiple frames.

    """
    energy = {}
    chemical_formula = {}
    is_lowest_energy = {}

    for (gIndexRun, gIndexDesc), atoms in structure.atoms.items():
        if atoms is not None:
            energy[gIndexRun, gIndexDesc] = structure.energy_eV[gIndexRun, gIndexDesc]
            chemical_formula[gIndexRun, gIndexDesc] = structure.chemical_formula[gIndexRun, gIndexDesc]

            lowest_energy_label_0 = dict_delta_e.get(chemical_formula[gIndexRun, gIndexDesc])[1]
            lowest_energy_label_1 = dict_delta_e.get(chemical_formula[gIndexRun, gIndexDesc])[2]

            if lowest_energy_label_0 > lowest_energy_label_1:
                lowest_energy_label_01 = lowest_energy_label_1
            else:
                lowest_energy_label_01 = lowest_energy_label_0

            if energy[gIndexRun, gIndexDesc] == lowest_energy_label_01:
                is_lowest_energy[gIndexRun, gIndexDesc] = True
            else:
                is_lowest_energy[gIndexRun, gIndexDesc] = False

    return is_lowest_energy


def write_atomic_features(structure, selected_feature_list, df, dict_delta_e=None,
                          path=None, filename_suffix='.json', json_file=None):
    """Given the chemical composition, build the descriptor made of atomic features only.

    Includes all the frames in the same json file.

    .. todo:: Check if it works for multiple frames.
    """

    # make dictionary {primary_feature: value} for each structure
    # dictionary of a dictionary, key: Mat, value: atomic_features
    dict_features = df.set_index('chemical_formula').T.to_dict()

    # label=0: rocksalt, label=1: zincblend
    #chemical_formula_, energy_, label_ = classify_rs_zb(structure)

    #is_lowest_energy_ = get_lowest_energy_structures(structure, dict_delta_e)

    if structure.isPeriodic == True:
        for (gIndexRun, gIndexDesc), atoms in structure.atoms.items():
            if atoms is not None:
                # filename is the normalized absolute path
                filename = os.path.abspath(os.path.normpath(os.path.join(path,
                                                                         '{0}{1}'.format(structure.name, filename_suffix))))

                outF = file(filename, 'w')
                outF.write("""
        {
              "data":[""")

                cell = structure.atoms[gIndexRun, gIndexDesc].get_cell()
                cell = np.transpose(cell)
                atoms = structure.atoms[gIndexRun, gIndexDesc]
                chemical_formula = structure.chemical_formula_[gIndexRun, gIndexDesc]
                energy = structure.energy_eV[gIndexRun, gIndexDesc]
                label = label_[gIndexRun, gIndexDesc]
                #target = dict_delta_e.get(chemical_formula_[gIndexRun, gIndexDesc])[0]
                target = dict_delta_e.get(chemical_formula)

                atomic_features = dict_features[structure.chemical_formula[gIndexRun, gIndexDesc]]
                #is_lowest_energy = is_lowest_energy_[gIndexRun,gIndexDesc]

                res = {
                    "checksum": structure.name,
                    "label": label,
                    "energy": energy,
                    #"is_lowest_energy": is_lowest_energy,
                    "delta_e_rs_zb": target,
                    "chemical_formula": chemical_formula,
                    "gIndexRun": gIndexRun,
                    "gIndexDesc": gIndexDesc,
                    "cell": cell.tolist(),
                    "particle_atom_number": map(lambda x: x.number, atoms),
                    "particle_position": map(lambda x: [x.x, x.y, x.z], atoms),
                    "atomic_features": atomic_features,
                    "main_json_file_name": json_file,
                }

                json.dump(res, outF, indent=2)
                outF.write("""
        ] }""")
                outF.flush()

    return filename


def r_sigma(row):
    """Calculates r_sigma.

    John-Bloch's indicator1: |rp(A) + rs(A) - rp(B) -rs(B)| from Phys. Rev. Lett. 33, 1095 (1974).

    Input rp(A), rs(A), rp(B), rs(B)
    They need to be given in this order.

    """
    return abs(row[0] + row[1] - row[2] + row[3])


def r_pi(row):
    """Calculates r_pi.

    John-Bloch's indicator2: |rp(A) - rs(A)| +| rp(B) -rs(B)| from Phys. Rev. Lett. 33, 1095 (1974).
    Input rp(A), rs(A), rp(B), rs(B)
    They need to be given in this order.
    combine_features
    """
    return abs(row[0] - row[1]) + abs(row[2] - row[3])


def e_sqrt_z(row):
    """Calculates e/sqrt(val_Z).

    Es/sqrt(Zval) and Ep/sqrt(Zval) from Phys. Rev. B 85, 104104 (2012).
    Input Es(A) or Ep(A), val(A)  (A-->B)
    They need to be given in this order.

    """
    return row[0] / math.sqrt(row[1])


def _get_scaling_factors(columns, metadata_info, energy_unit, length_unit):
    """Calculates characteristic energy and length, given an atomic metadata"""
    scaling_factor = []
    if columns is not None:
        for col in columns:
            try:
                col_unit = metadata_info[col.split('(', 1)[0]]['units']

                # check allowed values, to avoid problem with substance - NOT IDEAD
                if col_unit == 'J':
                    scaling_factor.append(uc.convert_unit(1, energy_unit, target_unit='eV'))
                    # divide all column by e_0
                    #df.loc[:, col] *= e_0
                elif col_unit == 'm':
                    scaling_factor.append(uc.convert_unit(1, length_unit, target_unit='angstrom'))
                    # divide all column by e_0
                    #df.loc[:, col] *= d_0
                else:
                    scaling_factor.append(1.0)
                    logger.debug("Feature units are not energy nor lengths. "
                                 "No scale to characteristic length.")
            except BaseException:
                scaling_factor.append(1.0)
                logger.debug("Feature units not included in metadata")

    return scaling_factor


def _my_power_2(row):
    return pow(row[0], 2)


def _my_power_3(row):
    return pow(row[0], 3)


def _my_power_m1(row):
    return pow(row[0], -1)


def _my_power_m2(row):
    return pow(row[0], -2)


def _my_power_m3(row):
    return pow(row[0], -3)


def _my_abs_sqrt(row):
    return math.sqrtabs(abs(row[0]))


def _my_exp(row):
    return exp(row[0])


def _my_exp_power_2(row):
    return exp(pow(row[0], 2))


def _my_exp_power_3(row):
    return exp(pow(row[0], 3))


def _my_sum(row):
    return row[0] + row[1]


def _my_abs_sum(row):
    return abs(row[0] + row[1])


def _my_abs_diff(row):
    return abs(row[0] - row[1])


def _my_diff(row):
    return row[0] - row[1]


def _my_div(row):
    return row[0] / row[1]


def _my_sum_power_2(row):
    return pow((row[0] + row[1]), 2)


def _my_sum_power_3(row):
    return pow((row[0] + row[1]), 3)


def _my_sum_exp(row):
    return exp(row[0] + row[1])


def _my_sum_exp_power_2(row):
    return exp(pow(row[0] + row[1], 2))


def _my_sum_exp_power_3(row):
    return exp(pow(row[0] + row[1], 3))


def combine_features(df=None, energy_unit=None, length_unit=None,
                     metadata_info=None, allowed_operations=None, derived_features=None):
    """Generate combination of features given a dataframe and a list of allowed operations.

    For the exponentials, we introduce a characteristic energy/length
    converting the
    ..todo:: Fix under/overflow errors, and introduce handling of exceptions.

    """

    if allowed_operations:
        logger.info('Selected operations:\n {0}'.format(allowed_operations))
    else:
        logger.warning('No allowed operations selected.')

    # make derived features
    if derived_features is not None:
        if 'r_sigma' in derived_features:
            # calculate r_sigma and r_pi [Phys. Rev. Lett. 33, 1095(1974)]
            logger.info('Including rs and rp to allow r_sigma calculation')
            radii_s_p = ['atomic_rp_max(A)', 'atomic_rs_max(A)', 'atomic_rp_max(B)', 'atomic_rs_max(B)']
            df['r_sigma'] = df[radii_s_p].apply(r_sigma, axis=1)

        if 'r_pi' in derived_features:
            logger.info('Including rs and rp to allow r_pi calculation')
            radii_s_p = ['atomic_rp_max(A)', 'atomic_rs_max(A)', 'atomic_rp_max(B)', 'atomic_rs_max(B)']
            df['r_pi'] = df[radii_s_p].apply(r_pi, axis=1)

    # calculate Es/sqrt(Zval) and Ep/sqrt(Zval)
#    e_val_z = ['Es(A)', 'val(A)']
#    df['Es(A)/sqrt(Zval(A))'] = df[e_val_z].apply(e_sqrt_z, axis=1)
#    e_val_z = ['Es(B)', 'val(B)']
#    df['Es(B)/sqrt(Zval(B))'] = df[e_val_z].apply(e_sqrt_z, axis=1)
#
#    e_val_z = ['Ep(A)', 'val(A)']
#    df['Ep(A)/sqrt(Zval(A))'] = df[e_val_z].apply(e_sqrt_z, axis=1)
#    e_val_z = ['Ep(B)', 'val(B)']
#    df['Ep(B)/sqrt(Zval(B))'] = df[e_val_z].apply(e_sqrt_z, axis=1)

    columns_ = df.columns.tolist()

    # define subclasses of features (see Phys. Rev. Lett. 114, 105503(2015) Supp. info. pag.1)
    # make a dictionary {feature: subgroup}
    # features belonging to a0 will not be combined, just added at the end

#    dict_features = {
#        u'val(B)': 'a0', u'val(A)': 'a0',
#
#        u'period__el0':'a0',
#        u'period__el1':'a0',
#        u'atomic_number__el0': 'a0',
#        u'atomic_number__el1': 'a0',
#        u'group__el0': 'a0',
#        u'group__el1': 'a0',
#
#        u'atomic_ionization_potential__el0': 'a1',
#        u'atomic_ionization_potential__el1': 'a1',
#        u'atomic_electron_affinity__el0': 'a1',
#        u'atomic_electron_affinity__el1': 'a1',
#        u'atomic_homo_lumo_diff__el0': 'a1',
#        u'atomic_homo_lumo_diff__el1': 'a1',
#        u'atomic_electronic_binding_energy_el0': 'a1',
#        u'atomic_electronic_binding_energy_el1': 'a1',
#
#
#        u'HOMO(A)': 'a2', u'LUMO(A)': 'a2', u'HOMO(B)': 'a2', u'LUMO(B)': 'a2',
#        u'HL_gap_AB': 'a2',
#        u'Ebinding_AB': 'a2',
#
#        u'atomic_rs_max__el0': 'a3',
#        u'atomic_rs_max__el1': 'a3',
#        u'atomic_rp_max__el0': 'a3',
#        u'atomic_rp_max__el1': 'a3',
#        u'atomic_rd_max__el0': 'a3',
#        u'atomic_rd_max__el1': 'a3',
#        u'atomic_r_by_2_dimer__el0': 'a3',
#        u'atomic_r_by_2_dimer__el1': 'a3',
#
#        u'd_AB': 'a3',
#        u'r_sigma': 'a3', u'r_pi': 'a3',
#
#        u'Eh': 'a4', u'C': 'a4'
#        }

    dict_features = {
        u'period': 'a0',
        u'atomic_number': 'a0',
        u'group': 'a0',

        u'atomic_ionization_potential': 'a1',
        u'atomic_electron_affinity': 'a1',
        u'atomic_homo_lumo_diff': 'a1',
        u'atomic_electronic_binding_energy': 'a1',

        u'atomic_homo': 'a2', u'atomic_lumo': 'a2',


        u'atomic_rs_max': 'a3',
        u'atomic_rp_max': 'a3',
        u'atomic_rd_max': 'a3',
        u'atomic_r_by_2_dimer': 'a3',

        u'r_sigma': 'a3', u'r_pi': 'a3'

    }

    # standardize the data -
    # we cannot reproduce the PRL if we standardize the data
    #df_a0 = (df_a0 - df_a0.mean()) / (df_a0.max() - df_a0.min())
    #df_a1 = (df_a1 - df_a1.mean()) / (df_a1.max() - df_a1.min())
    #df_a2 = (df_a2 - df_a2.mean()) / (df_a2.max() - df_a2.min())
    #df_a3 = (df_a3 - df_a3.mean()) / (df_a3.max() - df_a3.min())
    #df_a4 = (df_a4 - df_a4.mean()) / (df_a4.max() - df_a4.min())


#    df_a0 = df[[col for col in columns_ if dict_features.get(col)=='a0']].astype('float32')
    df_a0 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0]) == 'a0']].astype('float32')
    df_a1 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0]) == 'a1']].astype('float32')
    df_a2 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0]) == 'a2']].astype('float32')
    df_a3 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0]) == 'a3']].astype('float32')
    df_a4 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0]) == 'a4']].astype('float32')

    col_a0 = df_a0.columns.tolist()
    col_a1 = df_a1.columns.tolist()
    col_a2 = df_a2.columns.tolist()
    col_a3 = df_a3.columns.tolist()
    col_a4 = df_a4.columns.tolist()

    #  this list will at the end all the dataframes created
    df_list = []

    df_b0_list = []
    df_b1_list = []
    df_b2_list = []
    df_b3_list = []
    df_c3_list = []
    df_d3_list = []
    df_e3_list = []
    df_f1_list = []
    df_f2_list = []
    df_f3_list = []
    df_x1_list = []
    df_x2_list = []
    df_x_list = []

    # create b0: absolute differences and sums of a0
    # this is not in the PRL.
    for subset in itertools.combinations(col_a0, 2):
        if '+' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')']
            data = df_a0[list(subset)].apply(_my_sum, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if '-' in allowed_operations:
            cols = ['(' + subset[0] + '-' + subset[1] + ')']
            data = df_a0[list(subset)].apply(_my_diff, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

            cols = ['(' + subset[1] + '-' + subset[0] + ')']
            data = df_a0[list(subset)].apply(_my_diff, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if '|+|' in allowed_operations:
            cols = ['|' + subset[0] + '+' + subset[1] + '|']
            data = df_a0[list(subset)].apply(_my_abs_sum, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if '|-|' in allowed_operations:
            cols = ['|' + subset[0] + '-' + subset[1] + '|']
            data = df_a0[list(subset)].apply(_my_abs_diff, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if '/' in allowed_operations:
            cols = [subset[0] + '/' + subset[1]]
            data = df_a0[list(subset)].apply(_my_div, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

            cols = [subset[1] + '/' + subset[0]]
            data = df_a0[list(subset)].apply(_my_div, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a0, 1):
        if '^2' in allowed_operations:
            cols = [subset[0] + '^2']
            data = df_a0[list(subset)].apply(_my_power_2, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if '^3' in allowed_operations:
            cols = [subset[0] + '^3']
            data = df_a0[list(subset)].apply(_my_power_3, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

        if 'exp' in allowed_operations:
            cols = ['exp(' + subset[0] + ')']
            data = df_a0[list(subset)].apply(_my_exp, axis=1)
            df_b0_list.append(pd.DataFrame(data, columns=cols))

    # create b1: absolute differences and sums of a1
    for subset in itertools.combinations(col_a1, 2):
        if '+' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')']
            data = df_a1[list(subset)].apply(_my_sum, axis=1)
            df_b1_list.append(pd.DataFrame(data, columns=cols))

        if '-' in allowed_operations:
            cols = ['(' + subset[0] + '-' + subset[1] + ')']
            data = df_a1[list(subset)].apply(_my_diff, axis=1)
            df_b1_list.append(pd.DataFrame(data, columns=cols))

        if '|+|' in allowed_operations:
            cols = ['|' + subset[0] + '+' + subset[1] + '|']
            data = df_a1[list(subset)].apply(_my_abs_sum, axis=1)
            df_b1_list.append(pd.DataFrame(data, columns=cols))

        if '|-|' in allowed_operations:
            cols = ['|' + subset[0] + '-' + subset[1] + '|']
            data = df_a1[list(subset)].apply(_my_abs_diff, axis=1)
            df_b1_list.append(pd.DataFrame(data, columns=cols))

    # create b2: absolute differences and sums of a2
    for subset in itertools.combinations(col_a2, 2):
        if '+' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')']
            data = df_a2[list(subset)].apply(_my_sum, axis=1)
            df_b2_list.append(pd.DataFrame(data, columns=cols))

        if '-' in allowed_operations:
            cols = ['(' + subset[0] + '-' + subset[1] + ')']
            data = df_a2[list(subset)].apply(_my_diff, axis=1)
            df_b2_list.append(pd.DataFrame(data, columns=cols))

        if '|+|' in allowed_operations:
            cols = ['|' + subset[0] + '+' + subset[1] + '|']
            data = df_a2[list(subset)].apply(_my_abs_sum, axis=1)
            df_b2_list.append(pd.DataFrame(data, columns=cols))

        if '|-|' in allowed_operations:
            cols = ['|' + subset[0] + '-' + subset[1] + '|']
            data = df_a2[list(subset)].apply(_my_abs_diff, axis=1)
            df_b2_list.append(pd.DataFrame(data, columns=cols))

    # create b3: absolute differences and sums of a3
    for subset in itertools.combinations(col_a3, 2):
        if '+' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')']
            data = df_a3[list(subset)].apply(_my_sum, axis=1)
            df_b3_list.append(pd.DataFrame(data, columns=cols))

        if '-' in allowed_operations:
            cols = ['(' + subset[0] + '-' + subset[1] + ')']
            data = df_a3[list(subset)].apply(_my_diff, axis=1)
            df_b3_list.append(pd.DataFrame(data, columns=cols))

        if '|+|' in allowed_operations:
            cols = ['|' + subset[0] + '+' + subset[1] + '|']
            data = df_a3[list(subset)].apply(_my_abs_sum, axis=1)
            df_b3_list.append(pd.DataFrame(data, columns=cols))

        if '|-|' in allowed_operations:
            cols = ['|' + subset[0] + '-' + subset[1] + '|']
            data = df_a3[list(subset)].apply(_my_abs_diff, axis=1)
            df_b3_list.append(pd.DataFrame(data, columns=cols))

    # create c3: two steps:
    # 1) squares of a3 - unary operations
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        if '^2' in allowed_operations:
            cols = [subset[0] + '^2']
            data = df_a3[list(subset)].apply(_my_power_2, axis=1)
            df_c3_list.append(pd.DataFrame(data, columns=cols))
        if '^3' in allowed_operations:
            cols = [subset[0] + '^3']
            data = df_a3[list(subset)].apply(_my_power_3, axis=1)
            df_c3_list.append(pd.DataFrame(data, columns=cols))

    # 2) squares of b3 (only sums) --> sum squared of a3
    for subset in itertools.combinations(col_a3, 2):
        if '^2' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')^2']
            data = df_a3[list(subset)].apply(_my_sum_power_2, axis=1)
            df_c3_list.append(pd.DataFrame(data, columns=cols))

        if '^3' in allowed_operations:
            cols = ['(' + subset[0] + '+' + subset[1] + ')^3']
            data = df_a3[list(subset)].apply(_my_sum_power_3, axis=1)
            df_c3_list.append(pd.DataFrame(data, columns=cols))

    # create d3: two steps:
    # 1) exponentials of a3 - unary operations
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        if 'exp' in allowed_operations:
            cols = ['exp(' + subset[0] + ')']
            # find scaling factor for e_0 or d_0 for scaling
            # and multiply each column by the scaling factor
            scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
            df_subset = df_a3[list(subset)] * scaling_factors
            data = df_subset.apply(_my_exp, axis=1)
            df_d3_list.append(pd.DataFrame(data, columns=cols))

    # 2) exponentials of b3 (only sums) --> exponential of sum of a3
    for subset in itertools.combinations(col_a3, 2):
        if 'exp' in allowed_operations:
            cols = ['exp(' + subset[0] + '+' + subset[1] + ')']
            # find scaling factor for e_0 or d_0 for scaling
            # and multiply each column by the scaling factor
            scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
            df_subset = df_a3[list(subset)] * scaling_factors
            data = df_subset.apply(_my_sum_exp, axis=1)
            df_d3_list.append(pd.DataFrame(data, columns=cols))

    # create e3: two steps:
    # 1) exponentials of squared a3 - unary operations
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        operations = {'exp', '^2'}
        if operations <= set(allowed_operations):
            cols = ['exp(' + subset[0] + '^2)']
            # find scaling factor for e_0 or d_0 for scaling
            # and multiply each column by the scaling factor
            scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
            df_subset = df_a3[list(subset)] * scaling_factors
            data = df_subset.apply(_my_exp_power_2, axis=1)
            df_e3_list.append(pd.DataFrame(data, columns=cols))

        operations = {'exp', '^3'}
        if operations <= set(allowed_operations):
            try:
                cols = ['exp(' + subset[0] + '^3)']
                # find scaling factor for e_0 or d_0 for scaling
                # and multiply each column by the scaling factor
                scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
                df_subset = df_a3[list(subset)] * scaling_factors
                data = df_subset.apply(_my_exp_power_3, axis=1)
                df_e3_list.append(pd.DataFrame(data, columns=cols))
            except OverflowError as e:
                logger.warning('Dropping feature combination that caused under/overflow.\n')

    # 2) exponentials of b3 (only sums) --> exponential of sum of a3
    for subset in itertools.combinations(col_a3, 2):
        operations = {'exp', '^2'}
        if operations <= set(allowed_operations):
            cols = ['exp((' + subset[0] + '+' + subset[1] + ')^2)']
            # find scaling factor for e_0 or d_0 for scaling
            # and multiply each column by the scaling factor
            scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
            df_subset = df_a3[list(subset)] * scaling_factors
            data = df_subset.apply(_my_sum_exp_power_2, axis=1)
            df_e3_list.append(pd.DataFrame(data, columns=cols))

        operations = {'exp', '^3'}
        if operations <= set(allowed_operations):
            try:
                cols = ['exp((' + subset[0] + '+' + subset[1] + ')^3)']
                # find scaling factor for e_0 or d_0 for scaling
                # and multiply each column by the scaling factor
                scaling_factors = _get_scaling_factors(list(subset), metadata_info, energy_unit, length_unit)
                df_subset = df_a3[list(subset)] * scaling_factors
                data = df_subset.apply(_my_sum_exp_power_3, axis=1)
                df_e3_list.append(pd.DataFrame(data, columns=cols))
            except OverflowError as e:
                logger.warning('Dropping feature combination that caused under/overflow.\n')

    # make dataframes from lists, check if they are not empty
    # we make there here because they are going to be used to further
    # combine the features
    if not df_a0.empty:
        df_list.append(df_a0)

    if not df_a1.empty:
        df_x1_list.append(df_a1)
        df_list.append(df_a1)

    if not df_a2.empty:
        df_x1_list.append(df_a2)
        df_list.append(df_a2)

    if not df_a3.empty:
        df_x1_list.append(df_a3)
        df_list.append(df_a3)

    if not df_a4.empty:
        df_list.append(df_a4)

    if df_b0_list:
        df_b0 = pd.concat(df_b0_list, axis=1)
        col_b0 = df_b0.columns.tolist()
        df_b0.to_csv('./df_b0.csv', index=True)
        df_list.append(df_b0)

    if df_b1_list:
        df_b1 = pd.concat(df_b1_list, axis=1)
        col_b1 = df_b1.columns.tolist()
        df_x1_list.append(df_b1)
        df_list.append(df_b1)

    if df_b2_list:
        df_b2 = pd.concat(df_b2_list, axis=1)
        col_b2 = df_b2.columns.tolist()
        df_x1_list.append(df_b2)
        df_list.append(df_b2)

    if df_b3_list:
        df_b3 = pd.concat(df_b3_list, axis=1)
        col_b3 = df_b3.columns.tolist()
        df_x1_list.append(df_b3)
        df_list.append(df_b3)

    if df_c3_list:
        df_c3 = pd.concat(df_c3_list, axis=1)
        col_c3 = df_c3.columns.tolist()
        df_x2_list.append(df_c3)
        df_list.append(df_c3)

    if df_d3_list:
        df_d3 = pd.concat(df_d3_list, axis=1)
        col_d3 = df_d3.columns.tolist()
        df_x2_list.append(df_d3)
        df_list.append(df_d3)

    if df_e3_list:
        df_e3 = pd.concat(df_e3_list, axis=1)
        col_e3 = df_e3.columns.tolist()
        df_x2_list.append(df_e3)
        df_list.append(df_e3)

    if df_x1_list:
        df_x1 = pd.concat(df_x1_list, axis=1)
        col_x1 = df_x1.columns.tolist()

    if df_x2_list:
        df_x2 = pd.concat(df_x2_list, axis=1)
        col_x2 = df_x2.columns.tolist()

    # create f1 - abs differences and sums of b1 without repetitions
    # TO DO: calculate f1

    # create x - ratios of any of {a_i, b_i} i=1,2,3
    # with any of {c3, d3, e3} - typo in the PRL - no a3
    # total = (4+4+6+12+12+30)*(21+21+21) = 68*63 = 4284
    # for subset in itertools.combinations(col_a3, 1):
    #    if 'exp' in allowed_operations:
    #        cols = ['exp('+subset[0]+'^2)']
    #        data = df_a3[list(subset)].apply(_my_exp_power_2, axis=1)
    #        df_e3_list.append(pd.DataFrame(data, columns=cols))

    if df_x1_list and df_x2_list:
        for el_x1 in col_x1:
            for el_x2 in col_x2:
                if '/' in allowed_operations:
                    cols = [el_x1 + '/' + el_x2]
                    # now the operation is between two dataframes
                    data = df_x1[el_x1].divide(df_x2[el_x2])
                    df_x_list.append(pd.DataFrame(data, columns=cols))

    if df_f1_list:
        df_f1 = pd.concat(df_f1_list, axis=1)
        col_f1 = df_f1.columns.tolist()
        df_list.append(df_f1)

    if df_x_list:
        df_x = pd.concat(df_x_list, axis=1)
        col_x = df_x.columns.tolist()
        df_list.append(df_x)

    logger.debug('\n l1-l0 feature creation')

    if not df_a0.empty:
        logger.debug('Number of features in subgroup a0: {0}'.format(df_a0.shape[1]))
        logger.debug('Example of feature in subgroup a0: {0}'
                     .format(df_a0.columns.tolist()[random.randint(0, df_a0.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup a0.')

    if not df_a1.empty:
        logger.debug('Number of features in subgroup a1: {0}'.format(df_a1.shape[1]))
        logger.debug('Example of feature in subgroup a1: {0}'
                     .format(df_a1.columns.tolist()[random.randint(0, df_a1.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup a1.')

    if not df_a2.empty:
        logger.debug('Number of features in subgroup a2: {0}'.format(df_a2.shape[1]))
        logger.debug('Example of feature in subgroup a2: {0}'
                     .format(df_a2.columns.tolist()[random.randint(0, df_a2.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup a2.')

    if not df_a3.empty:
        logger.debug('Number of features in subgroup a3: {0}'.format(df_a3.shape[1]))
        logger.debug('Example of feature in subgroup a3: {0}'
                     .format(df_a3.columns.tolist()[random.randint(0, df_a3.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup a3.')

    if not df_a4.empty:
        logger.debug('Number of features in subgroup a4: {0}'.format(df_a4.shape[1]))
        logger.debug('Example of feature in subgroup a4: {0}'
                     .format(df_a3.columns.tolist()[random.randint(0, df_a4.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup a4.')

    if df_b0_list:
        logger.debug('Number of features in subgroup b0: {0}'.format(df_b0.shape[1]))
        logger.debug('Example of feature in subgroup b0: {0}'
                     .format(df_b0.columns.tolist()[random.randint(0, df_b0.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup b0.')

    if df_b1_list:
        logger.debug('Number of features in subgroup b1: {0}'.format(df_b1.shape[1]))
        logger.debug('Example of feature in subgroup b1: {0}'
                     .format(df_b1.columns.tolist()[random.randint(0, df_b1.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup b1.')

    if df_b2_list:
        logger.debug('Number of features in subgroup b2: {0}'.format(df_b2.shape[1]))
        logger.debug('Example of feature in subgroup b2: {0}'
                     .format(df_b2.columns.tolist()[random.randint(0, df_b2.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup b2.')

    if df_b3_list:
        logger.debug('Number of features in subgroup b3: {0}'.format(df_b3.shape[1]))
        logger.debug('Example of feature in subgroup b3: {0}'
                     .format(df_b3.columns.tolist()[random.randint(0, df_b3.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup b3.')

    if df_c3_list:
        logger.debug('Number of features in subgroup c3: {0}'.format(df_c3.shape[1]))
        logger.debug('Example of feature in subgroup c3: {0}'
                     .format(df_c3.columns.tolist()[random.randint(0, df_c3.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup c3.')

    if df_d3_list:
        logger.debug('Number of features in subgroup d3: {0}'.format(df_d3.shape[1]))
        logger.debug('Example of feature in subgroup d3: {0}'
                     .format(df_d3.columns.tolist()[random.randint(0, df_d3.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup d3.')

    if df_e3_list:
        logger.debug('Number of features in subgroup e3: {0}'.format(df_e3.shape[1]))
        logger.debug('Example of feature in subgroup e3: {0}'
                     .format(df_e3.columns.tolist()[random.randint(0, df_e3.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup e3.')

    if df_f1_list:
        logger.debug('Number of features in subgroup f1: {0}'.format(df_f1.shape[1]))
        logger.debug('Example of feature in subgroup f1: {0}'
                     .format(df_f1.columns.tolist()[random.randint(0, df_f1.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup f1.')

    if df_x_list:
        logger.debug('Number of features in subgroup x: {0}'.format(df_x.shape[1]))
        logger.debug('Example of feature in subgroup x: {0}'
                     .format(df_x.columns.tolist()[random.randint(0, df_x.shape[1] - 1)]))
    else:
        logger.debug('No features in subgroup x.')

    logger.debug('Please see Phys. Rev. Lett. 114, 105503(2015) Supplementary Information \n for more details.\n')

    if df_list:
        df_combined_features = pd.concat(df_list, axis=1)
    else:
        logger.error('No features selected. Please select at least two primary features.')
        sys.exit(1)

    logger.info('Number of total features generated: {0}'.format(df_combined_features.shape[1]))

    return df_combined_features


def l1_l0_minimization(y_true, D, features,
                       energy_unit=None,
                       print_lasso=False, lambda_grid=None, lassonumber=25,
                       max_dim=3, lambda_grid_points=100, lambda_max_factor=1.0, lambda_min_factor=0.001):
    """ Select an optimal descriptor using a combined l1-l0 procedure.

    1. step (l 1): Solve the LASSO minimization problem

    .. math::
        argmin_c {||P-Dc||^2 + \lambda |c|_1}

    for different lambdas, starting from a 'high' lambda.
    Collect all indices(Features) i appearing with nonzero coefficients c_i,
    while decreasing lambda, until size of collection equals `lassonumber`.

    2. step (l 0): Check the least-squares errors for all single features/pairs/triples/... of
                   collection from 1. step. Choose the single/pair/triple/... with the lowest
                   mean squared error (MSE) to be the best 1D/2D/3D-descriptor.


    Parameters:

    y_true : array, [n_samples]
        Array with the target property (ground truth)

    D : array, [n_samples, n_features]
        Matrix with the data.

    features : list of strings
        List of feature names. Needs to be in the same order as the feature vectors in D

    dimrange : list of int
        Specify for which dimensions the optimal descriptor is calculated.
        It is the number of feature vectors used in the linear combination

    lassonumber : int, default 25
        The number of features, which will be collected in ther l1-step

    lamdba_grid_points : int, default 100
        Number of lamdbas between lamdba_max and lambdba_min for which the l1-problem shall be solved.
        Sometimes a denser grid could be needed, if the lamda-steps are too high.
        This can be checked with 'print_lasso'. `lamdba_max` and `lamdba_min` are chosen as in
        Tibshirani's paper "Regularization Paths for Generalized Linear Models via Coordinate Descent".
        The values in between are generated on the log scale.

    lambda_min_factor : float, default 0.001
        Sets `lam_min` = `lambda_min_factor` * `lam_max`.

    lambda_max_factor : float, default 1.0
        Sets calculated `lam_max` = `lam_max` * `lambda_max_factor`.

    print_lasso: bool, default `True`
        Prints the indices of coulumns of `D` with nonzero coefficients for each lambda.

    lambda_grid: array
        The list/array of lambda values for the l1-problem can be chosen by the user.
        The list/array should start from the highest number and lambda_i > lamda_i+1 should hold.
        (?) `lambda_grid_point` is then ignored. (?)

    Returns:

    list of panda dataframes
    (D', c', selected_features) :
        A list of tuples (D',c',selected_features) for each dimension.
        `selected_features` is a list of strings. D'*c' is the selected linear model/fit where the last column
        of `D` is a vector with ones.


    References:

    .. [1] Luca M. Ghiringhelli, Jan Vybiral, Sergey V. Levchenko, Claudia Draxl, and Matthias Scheffler,
        "Big Data of Materials Science: Critical Role of the Descriptor"
        Phys. Rev. Lett. 114, 105503 (2015)


    """

    dimrange = range(1, max_dim + 1)
    compounds = len(y_true)

    # standardize D
    Dstan = np.array(scipy.stats.zscore(D))

    y_true = y_true.flatten()

    #lambda_grid=[pow(1.7,i) for i in np.arange(-40.,-1,1.)]
    # lambda_grid.sort(reverse=True)

    logger.info('Selecting optimal descriptors.')

    if lambda_grid is None:
            # find max lambda, and build lambda grid as in
            # Tibshirani's paper "Regularization Paths for Generalized Linear
            # Models via Coordinate Descent". Here lam_max can be set to a higher
            # with a factor lambda_max_factor
        correlations = abs(np.dot(y_true, Dstan))
        correlations = np.asarray(correlations)

        lam_max = max(correlations) / (compounds)
        lam_min = lam_max * lambda_min_factor
        lam_max = lambda_max_factor * lam_max
        log_max, log_min = np.log10(lam_max), np.log10(lam_min)

        lambda_grid = [pow(10, i) for i in np.linspace(log_min, log_max, lambda_grid_points)]
        lambda_grid.sort(reverse=True)

    # LASSO begin, iter over lamda grid, and collect all indices(Features)
    # with nonzero coefficient until len(collection)=lassonumber
    collection = []
    if print_lasso:
        logger.debug('lambda      #collected   Indices')
    for l, lam in enumerate(lambda_grid):
        lasso = linear_model.Lasso(alpha=lam, copy_X=True, fit_intercept=True,
                                   max_iter=100000, normalize=False, positive=False, precompute=False,
                                   random_state=None, selection='cyclic', tol=0.0001, warm_start=False)
        lasso.fit(Dstan, y_true)
        coef = lasso.coef_
        for pos in np.nonzero(coef)[0]:
            if not pos in collection:
                collection.append(pos)
        if print_lasso:  # print the indices of nonzero coefficients for a given lambda.
                            # (It is NOT the collection at that moment)
            logger.debug('%.10f   %s   %s' % (lam, len(collection), np.nonzero(coef)[0]))
        if len(collection) > lassonumber - 1:
            break
    collection = sorted(collection[:lassonumber])
    # LASSO end

    # collection is the list with the features that have been collected
    len_collection = len(collection)
    if len_collection < lassonumber:
        logger.debug("Only %s features are collected" % len_collection)
    # make small matrix with size of (compounds,lassonumber), only with  selected features from LASSO
    D_collection = D[:, collection]
    D_collection = np.column_stack((D_collection, np.ones(compounds)))

    # get the different dimensional descriptor and save the
    # tuple (D_model, coefficients, selected_features) for each dimension in the list out
    out = []
    out_df = []
    y_pred = []

    for dimension in dimrange:

        # L0: save for each single Feature/ pair, triple/... the Least-Squares-Error
        # with its coefficient and index in Dictionary MSEdic
        MSEdic = {}
        for permu in itertools.combinations(range(len_collection), dimension):
            D_ls = D_collection[:, permu + (-1,)]
            x = np.linalg.lstsq(D_ls, y_true, rcond=None)
            # if there are linear dependencies in D_ls np.linalg.lstsq gives no error and len(x[1])==0.
            if not len(x[1]) == 0:
                                   # (There could be other reasons, too...which should/could be checked)
                MSE = x[1][0] / compounds
                MSEdic.update({MSE: [x[0], permu]})

        # check if MSEdic is empty
        if not bool(MSEdic):
            logger.error('Could not find configuration with lowest MSE.\n Try to select ' +
                         'more features\n or reduce the number of the descriptor dimension. ')
            sys.exit(1)

        # select the model with the lowest MSE
        minimum = min(MSEdic)

        logger.info("Root Mean Squared Error (RMSE) for {0}D descriptor: {1:.6e} {2}"
                    .format(dimension, sqrt(minimum), energy_unit))
        model = MSEdic[minimum]
        coefficients, good_permu = model[0], model[1]

        # transform the D_collection-indices into D-indices, and get strings for selected features
        # from the list 'features' , and get D_model
        selected_features = [features[collection[gp]] for gp in good_permu]
        D_model = D_collection[:, good_permu + (-1,)]

        # save the model for the actual dimension and strings of features
        out.append((D_model, coefficients, selected_features))

        # print in terminal
        string = '{0}D case: \n'.format(dimension)
        for i in range(dimension + 1):
            if coefficients[i] > 0:
                sign = '+'
                c = coefficients[i]
            else:
                sign = '-'
                c = abs(coefficients[i])
            if i < dimension:
                string += '%s %.6e %s\n ' % (sign, c, selected_features[i])
            else:
                string += '%s %.6e\n' % (sign, c)
        logger.info(string)

        # calculate E_predict
        y_pred.append(np.dot(D_model, coefficients))

        # RMSE (it should be the same as before): sqrt(mean_squared_error(P, P_pred))

        # create panda dataframe
        selected_features.append('Intercept')
        data = D_model

        out_df.append(pd.DataFrame(data, columns=selected_features))

    return out_df, y_pred, y_true
