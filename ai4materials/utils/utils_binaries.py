from ase.data import chemical_symbols
from ase.spacegroup import get_spacegroup
import logging
import numpy as np
import pandas as pd
from itertools import permutations
logger = logging.getLogger('ai4materials')


def get_target_diff_dic(df, sample_key=None, energy=None, spacegroup=None):
    """ Get a dictionary of dictionaries: samples -> space group tuples -> energy differences.

    Dropping all rows which do not correspond to the minimum energy per sample AND space group,
    then making a new data frame with space groups as columns. Finally constructing the dictionary
    of dictionaries.

    Parameters:

    df: pandas data frame
        with columns=[samples_title, energies_title, SG_title]

    sample_key: string
        Needs to be column title of samples of input df

    energy: string
        Needs to be column title of energies of input df

    spacegroup : string
        Needs to be column title of space groups of input df

    Returns:

    dic_out: dictionary of dictionaries:
        In the form:
        {
        sample_a: { (SG_1,SG_2):E_diff_a12, (SG_1,SG_3):E_diff_a13,...},
        sample_b: { (SG_1,SG_2):E_diff_b12, (SG_1,SG_3):E_diff_b13,... },
        ...
        }
        E_diff_a12 = energy_SG_1 - energy_SG_2   of sample a.
        Both (SG_1,SG_2) and (SG_2,SG_1) are considered.
        If SG_1 or SG_2 is NaN, energy difference to it is ignored.


    """

    # use only rows with minimum energies
    idx = df.groupby([sample_key, spacegroup])[energy].transform(min) == df[energy]
    df = df[idx]
    df = df.drop_duplicates()

    # make new table with the different supgroups as columns
    df = df.pivot_table(energy, [sample_key], spacegroup)

    # make dictionary of dictionaries
    SG_list = df.columns.values
    Samples_list = df.index.values
    matrix = np.array(df)
    dic_out = dict.fromkeys(Samples_list)
    for i, sample in enumerate(Samples_list):
        row = matrix[i]
        not_nan_indices = np.argwhere(~np.isnan(row)).flatten()
        sample_dic = {}
        for j_1, j_2 in permutations(not_nan_indices, 2):
            SG_1, SG_2 = SG_list[j_1], SG_list[j_2]
            Energy_diff = row[j_1] - row[j_2]
            sample_dic.update({(SG_1, SG_2): Energy_diff})
        if sample_dic:
            dic_out[sample] = sample_dic
    return dic_out


def select_diff_from_dic(dic, spacegroup_tuples, sample_key='Mat', drop_nan=None):
    """ Get data frame of selected spacegroup_tuples from dictionary of dictionaries.

        Creating a pandas data frame with columns of samples and selected space group tuples (energy differnces).

        Parameters:

        dic: dict {samples -> space group tuples -> energy differences.}

        spacegroup_tuples: tuple, list of tuples, tuples of tuples
            Each tuple has to contain two space groups numbers,
            to be looked up in the input dic.

        sample_key: string
            Will be the column title of the samples of the created data frame

        drop_nan: string, optional {'rows', 'SG_tuples'}
            Drops all rows or columns (SG_tuples) containing NaN.

    """

    if isinstance(spacegroup_tuples, tuple) and all(isinstance(item, (float, int)) for item in spacegroup_tuples):
        spacegroup_tuples = [spacegroup_tuples]
    df_out = pd.DataFrame(dic, index=spacegroup_tuples).T

    if not drop_nan is None:
        if drop_nan == 'rows':
            df_out.dropna(axis=0, inplace=True)
        elif drop_nan == 'SG_tuples':
            df_out.dropna(axis=1, inplace=True)
        else:
            raise ValueError("Argument 'drop_nan' has to be 'None', 'rows' or 'SG_tuples'.")

    # check if df_out is empty
    len_columns = len(df_out.columns)
    len_rows = len(df_out.index)
    if len_columns == 0 or len_rows == 0:
        if len_rows == 0:
            string = 'rows'
        else:
            string = 'spacegroup_tuples'
        logger.error('Dropping {0} with NaNs leads to empty data frame.'.format(string))
        logger.error('Hint: Select different spacegroup_tuples or set drop_nan=None')
        sys.exit(1)
    df_out.reset_index(inplace=True)
    df_out.rename(columns={'index': sample_key}, inplace=True)
    return df_out


def get_chemical_formula_binaries(atoms):

    numbers = atoms.get_atomic_numbers()
    elements = np.unique(numbers)
    symbols = np.array([chemical_symbols[e] for e in elements])

    ind = symbols.argsort()
    symbols = symbols[ind]

    if 'H' in symbols:
        i = np.arange(len(symbols))[symbols == 'H']
        symbols = np.insert(np.delete(symbols, i), 0, symbols[i])
    if 'C' in symbols:
        i = np.arange(len(symbols))[symbols == 'C']
        symbols = np.insert(np.delete(symbols, i), 0, symbols[i])

    formula = "".join(symbols)

    if len(symbols) == 1:
        formula += '2'

    return formula


def get_binaries_dict_delta_e(chemical_formula_list, energy_list, label_list, equiv_spgroups):
    energy_list = [item * 0.5 for item in energy_list]

    # make dataframe with chemical formula, energy, and labels
    data = zip(chemical_formula_list, energy_list, label_list)
    df_energy_diff = pd.DataFrame.from_records(data, columns=['chemical_formula', 'energy_total', 'spacegroup'])
    sample_key, energy, spacegroup = 'chemical_formula', 'energy_total', 'spacegroup'
    drop_nan = None  # or 'rows' or 'SG_tuples'

    selected_spacegroup_tuples, spacegroups_replace = zip(*equiv_spgroups)
    # replace space groups such that only one space group per structure is present
    df_energy_diff[spacegroup] = df_energy_diff[spacegroup].replace(spacegroups_replace, selected_spacegroup_tuples)

    target_diff_dic = get_target_diff_dic(
        df_energy_diff,
        sample_key=sample_key,
        energy=energy,
        spacegroup=spacegroup)

    target_df = select_diff_from_dic(
        target_diff_dic,
        selected_spacegroup_tuples,
        sample_key=sample_key,
        drop_nan=drop_nan)

    df_with_e_diff = df_energy_diff.merge(target_df, left_on=sample_key, right_on=sample_key)
    dict_delta_e = df_with_e_diff.set_index(sample_key)[selected_spacegroup_tuples].to_dict()

    return dict_delta_e


def get_energy_diff_by_spacegroup(ase_atoms_list, target='energy_total', equiv_spgroups=None):

    logging.debug("Using {} as target.".format(target))

    chemical_formula_list = []
    energy_list = []
    label_list = []

    for idx_atoms, ase_atoms in enumerate(ase_atoms_list):
        energy = ase_atoms.info[target]
        # get chemical_formula, energy, classification (space group) for binaires
        label = get_spacegroup(ase_atoms).no
        # chemical_formula = list(set(ase_atoms.get_chemical_symbols()))
        chemical_formula = ase_atoms.get_chemical_formula(mode='hill')

        chemical_formula_list.append(chemical_formula)
        label_list.append(label)
        energy_list.append(energy)

        # the last iteration calculate the energy differences between space group with all the json_file data

    dict_delta_e = get_binaries_dict_delta_e(
        chemical_formula_list,
        energy_list,
        label_list,
        equiv_spgroups=equiv_spgroups)

    return dict_delta_e
