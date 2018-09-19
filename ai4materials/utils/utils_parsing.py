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

import os
import logging
import numpy as np
import pandas as pd
import os.path
import pymatgen as mg
import ase
from ase.spacegroup import get_spacegroup
from pymatgen.io.ase import AseAtomsAdaptor
from ai4materials.utils.utils_crystals import get_spacegroup_analyzer
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from future.utils import viewitems

logger = logging.getLogger('ai4materials')


def read_gdb_7k(dataset=None, xyz_file=None, target_file=None):
    """ Read the gdb_7k dataset.

    Link to reference: http://iopscience.iop.org/article/10.1088/1367-2630/15/9/095003/meta;jsessionid=BE06A5911A9B58D271F6E2C145D54FDC.c2.iopscience.cld.iop.org
    Link to dataset: http://www.mrupp.info/Data/2013dsgdb7njp.zip

    Parameters:

    dataset: string
        File containing the dataset. If taken from the link above, it is named "dsgdb7njp.xyz".

    xyz_file: string, optional, default ./dsgdb7njp_ase.xyz
        File containing the configurations in the dataset that can be read by the ASE python library.

    target_file: string, optional, default ./dsgdb7njp.csv
        CSV file containing the target values as panda dataframe. See the code below for details.

    Returns:

    dataset: dataframe
         Dataframe containing the data read from the datasets. It is 7211 rows.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if xyz_file is None:
        xyz_file = './dsgdb7njp_ase.xyz'
        import pymatgen as mg

    if target_file is None:
        target_file = './dsgdb7njp.csv'

    logger.info('Preprocessing gdb13_7k database')

    i = 0
    f = open(dataset, 'r')
    out = open(xyz_file, 'w')
    l = [l for l in f.readlines()]
    # to read the 1st line correctly
    id_empty = [-1]

    for idx, l_ in enumerate(l):
        if not l_.strip():
            i += 1
            id_empty.append(idx)

    target_values = [l[idx + 2] for idx in id_empty]

    data = []

    for target_ in target_values:
        data.append(str(target_).split())

    labels = []
    # ae_pbe0    kcal/mol   Atomization energy (DFT/PBE0)
    labels.append('ae_pbe0')
    # p_pbe0     Angstrom^3 Polarizability (DFT/PBE0)
    labels.append('p_pbe0')
    # p_scs      Angstrom^3 Polarizability (self-consistent screening)
    labels.append('p_scs')
    # homo_gw    eV         Highest occupied molecular orbital (GW)
    labels.append('homo_gw')
    # homo_pbe0  eV         Highest occupied molecular orbital (DFT/PBE0)
    labels.append('homo_pbe0')
    # homo_zindo eV         Highest occupied molecular orbital (ZINDO/s)
    labels.append('homo_zindo')
    # lumo_gw    eV         Lowest unoccupied molecular orbital (GW)
    labels.append('lumo_gw')
    # lumo_pbe0  eV         Lowest unoccupied molecular orbital (DFT/PBE0)
    labels.append('lumo_pbe0')
    # lumo_zindo eV         Lowest unoccupied molecular orbital (ZINDO/s)
    labels.append('lumo_zindo')
    # ip_zindo   eV         Ionization potential (ZINDO/s)
    labels.append('ip_zindo')
    # ea_zindo   eV         Electron affinity (ZINDO/s)
    labels.append('ea_zindo')
    # e1_zindo   eV         First excitation energy (ZINDO)
    labels.append('e1_zindo')
    # emax_zindo eV         Maximal absorption intensity (ZINDO)
    labels.append('emax_zindo')
    # imax_zindo arbitrary  Excitation energy at maximal absorption (ZINDO)
    labels.append('imax_zindo')

    data = np.asarray(data).astype(float)
    df = pd.DataFrame(data=data, columns=labels)

    # add homo-lumo gaps
    df['homo_lumo_gap_gw'] = df['lumo_gw'] - df['homo_gw']
    df['homo_lumo_gap_pbe0'] = df['lumo_pbe0'] - df['homo_pbe0']

    # copy in the xyz file the original dataset without the empty lines
    lines = [line for line in open(dataset) if line[:-1]]
    f.close()
    out.writelines(lines)
    out.close()

    if df.shape[0] != 7211:
        logger.warning('The target value array should contain 7211 elements.')
        logger.warning('Your target value contains {0} elements.'.format(df.shape[0]))

    df.to_csv(target_file, index=False)
    logger.debug('Printing some statistics on the dataset')
    logger.debug(df.describe())

    logger.info('Preprocessing: done.')

    return dataset


def read_data(json_list, calc_spgroup=True, symprec=1e-03):
    """Read json file from the NOMAD Archive format and get a list of ASE structures"""

    frame_list = None
    frame_list_idcs = [(0, 0)]

    data_file_format = 'NOMAD'

    op_list = np.zeros(len(json_list))

    logger.info("Converting data (%d archives)..." % len(json_list))
    ase_atoms_list = []
    nmd_struct_list = []
    frame_list_idx_list = []
    target_list = []
    label_list = []
    z_count_global = {}
    for json_idx, json_file in enumerate(json_list):
        if json_idx % (int(len(json_list) / 10) + 1) == 0:
            logger.info("Reading: file {0}/{1}".format(json_idx + 1, len(json_list)))

        nmd_struct = NOMADStructure(in_file=json_file, frame_list=frame_list, file_format=data_file_format)
        frame_list_idx_list.append([])
        for idx in frame_list_idcs:
            ase_atoms = nmd_struct.atoms[idx]
            ase_atoms_list.append(ase_atoms)
            frame_list_idx_list[-1].append(idx[1])
        nmd_struct_list.append(nmd_struct)

    # name is filename without extension
    ase_atoms_names = [os.path.splitext(os.path.basename(item))[0] for item in json_list]

    # probably better to use a unique hash instead.
    for idx, ase_atoms in enumerate(ase_atoms_list):
        ase_atoms.info['idx'] = idx
        ase_atoms.info['label'] = ase_atoms_names[idx] + '_' + str(ase_atoms.info['idx'])

        if calc_spgroup:
            ase_atoms.info['spacegroup_nb'] = {}
            space_group_analyzer = get_spacegroup_analyzer(ase_atoms, symprec=symprec)
            for key, value in space_group_analyzer.items():
                ase_atoms.info['spacegroup_nb'][str(key)] = value.get_space_group_number()

    return ase_atoms_list


def get_pymatgen_ase_from_cif_structure(path):
    # warnings.filterwarnings('ignore')#, category=DeprecationWarning)
    struct = mg.Structure.from_file(path)
    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(struct)
    except Exception as e:
        print(repr(e))
    return ase_atoms


def read_atomic_structures(data_folder, nb_max_folders=1000, nb_max_files=10000, filename_suffix='.aims',
                           format_input='aims', calc_spgroup=False, symprec=(1e-03, 1e-06, 1e-09)):
    """Read geometry files from data_folder, and return ASE list"""

    ase_atoms_list = []

    i = 0
    j = 0
    for root, dirs, files in os.walk(data_folder):
        i += 1
        if i > nb_max_folders:  # this is for folders
            break
        for file_ in files:
            j += 1
            if j > nb_max_files:  # this is for files
                i = nb_max_folders + 1
                break
            if file_.endswith(filename_suffix):
                filepath = os.path.join(root, file_)
                filename_no_ext, file_extension = os.path.splitext(filepath)
                # read only first element
                atoms = ase.io.read(filepath, index=0, format=format_input)

                if calc_spgroup:
                    spgroups = [get_spacegroup(atoms, symprec=item).no for item in symprec]

                    for idx_spgroup, spgroups in enumerate(spgroups):
                        atoms.info['spacegroup_' + str(symprec[idx_spgroup])] = spgroups

                ase_atoms_list.append(atoms)

    return ase_atoms_list


def nmd_uri_to_ase_atoms_tmp(nmd_uris, get_energy_total=True, calc_spacegroup=True):
    """Temporary function to build ASE atoms from nmd uri. Will be substituted from the new option in nomad resolve."""

    from nomadcore.nomad_query import NomadQuery
    nomad_query = NomadQuery()

    ase_atoms_list = []
    target_list = []

    if get_energy_total:
        energy_total_list = []
        sec_retrieval_energy = {'section-2': dict(section_name='section_single_configuration_calculation', paths=[
            '/section_run/0c/section_single_configuration_calculation/0c'])}

        # for key, section in sec_retrieval.iteritems():
        for (key, section) in viewitems(sec_retrieval_energy):
            # define a default for retrieved keys
            # so it will not fail if nothing can be retrieved
            for idx_nmd, nmd in enumerate(nmd_uris):
                if idx_nmd % 10 == 0:
                    logger.debug('{0}/{1}'.format(idx_nmd + 1, len(nmd_uris)))

                for path in section['paths']:
                    nmd_dict = nomad_query.resolve(nmd=nmd, path=path, recursive=True)
                    energy_total = nmd_dict['energy_total']
                    print(nmd)
                energy_total_list.append(energy_total)

    sec_retrieval = {'section-1': {'section_name': 'section_system',
                                   'paths': ['/section_run/0c/section_system/0c', '/section_run/0c/section_system/1c'],
                                   'required_keys_atom_species': ['atom_positions', 'atom_species',
                                                                  'configuration_periodic_dimensions'],
                                   'required_keys_atom_labels': ['atom_positions', 'atom_labels',
                                                                 'configuration_periodic_dimensions']}}

    # for key, section in sec_retrieval.iteritems():
    for (key, section) in viewitems(sec_retrieval):
        # define a default for retrieved keys
        # so it will not fail if nothing can be retrieved
        required_keys = section['required_keys_atom_species']

        for idx_nmd, nmd in enumerate(nmd_uris):
            retrieval_success = False
            if idx_nmd % 10 == 0:
                logger.debug('{0}/{1}'.format(idx_nmd + 1, len(nmd_uris)))

            for path in section['paths']:
                try:
                    nmd_dict = nomad_query.resolve(nmd=nmd, path=path, recursive=True)
                except BaseException:
                    nmd_dict = []

                # first try with atom species (this is the correct procedure)
                if all([required_key in nmd_dict for required_key in section['required_keys_atom_species']]):
                    logging.debug("Using path {} for query-data retrieval".format(path))
                    retrieval_success = True
                    required_keys = section['required_keys_atom_species']
                    # logger.debug("Retrieval succedeed: {}".format(retrieval_success))
                    # logger.debug("Retrieved keys: {}".format(retrieved_keys))
                    break

            if not retrieval_success:
                # try with atom labels (this is not the correct procedure because atom_labels do not have
                # necessary to be atomic species (e.g. C1, C2)
                # we need to do this because there was a bug in the Gaussian parser
                for path in section['paths']:
                    try:
                        nmd_dict = nomad_query.resolve(nmd=nmd, path=path, recursive=True)
                    except BaseException:
                        nmd_dict = []

                    # first try with atom species (this is the correct procedure)
                    if all([required_key in nmd_dict for required_key in section['required_keys_atom_labels']]):
                        logging.debug("Using path {} for query-data retrieval".format(path))
                        required_keys = section['required_keys_atom_labels']
                        # logger.debug("Retrieved keys: {}".format(retrieved_keys))
                        break

            if all([required_key in nmd_dict for required_key in required_keys]):
                if section['section_name'] == 'section_system':
                    atom_positions = np.array(nmd_dict['atom_positions']['flatData']).reshape(
                        nmd_dict['atom_positions']['shape']) * 1.0E+10
                    periodicity = np.array(nmd_dict['configuration_periodic_dimensions'][0]['flatData'], dtype=bool)

                    if required_keys == section['required_keys_atom_species']:
                        atom_species = np.array(nmd_dict['atom_species'])
                    elif required_keys == section['required_keys_atom_labels']:
                        atom_species = np.array(nmd_dict['atom_labels']['flatData'])
                    else:
                        logger.error("No retrieved keys. Stopping.")

                    # read cell only if periodicity is all True
                    # the Gaussian parser for non-periodic systems still the cell (0., 0., 0.) even if there is no cell
                    if np.all(periodicity):
                        try:
                            simulation_cell = np.array(nmd_dict['simulation_cell']['flatData']).reshape(
                                nmd_dict['simulation_cell']['shape']) * 1.0E+10
                        except BaseException:
                            simulation_cell = None
                    else:
                        logger.debug("Calculation {} is not periodic over three dimensions.".format(nmd))

                    # create ASE object
                    if np.all(periodicity):
                        atoms = ase.Atoms(symbols=atom_species, positions=atom_positions, cell=simulation_cell,
                                          pbc=True)

                        # calculate spacegroup on-the-fly
                        # use spacegroup as target for crystals
                        if calc_spacegroup:
                            try:
                                spgroup_nb = get_spacegroup(atoms, symprec=0.001).no
                            except BaseException:
                                spgroup_nb = None

                        atoms.info['spgroup_nb'] = spgroup_nb
                        atoms.info['target'] = spgroup_nb

                    else:
                        atoms = ase.Atoms(symbols=atom_species, positions=atom_positions, cell=simulation_cell,
                                          pbc=False)

                        # calculate pointgroup on-the-fly
                        # use pointgroup as target for molecules from pymatgen
                        if calc_spacegroup:
                            try:
                                point_group_symbol = PointGroupAnalyzer(AseAtomsAdaptor.get_structure(atoms),
                                                                        symprec=0.001).get_point_group_symbol()
                            except BaseException:
                                point_group_symbol = None

                        atoms.info['point_group_symbol'] = point_group_symbol
                        atoms.info['target'] = point_group_symbol

                    # add label
                    atoms.info['nmd_uri'] = nmd
                    atoms.info['nmd_checksum'] = nmd.rsplit('/')[-1]

                    # set label
                    # using NMD uri is better because unique, but it prints more to screen
                    atoms.info['label'] = nmd.rsplit('/')[-1]
                    # atoms.info['label'] = str(idx_nmd)

                    ase_atoms_list.append(atoms)
                    target_list.append(atoms.info['target'])

                else:
                    pass

            else:
                logger.info(
                    "Could not retrieve atom_positions and/or atom_species for entry: {} in paths {}".format(nmd,
                                                                                                             section[
                                                                                                                 'paths']))

    if get_energy_total:
        assert (len(ase_atoms_list) == len(energy_total_list))

        for idx, atoms in enumerate(ase_atoms_list):
            atoms.info['energy_total'] = energy_total_list[idx]

    return ase_atoms_list
