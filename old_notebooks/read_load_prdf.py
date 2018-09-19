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
__copyright__ = "Copyright 2016-2018, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/04/18"

if __name__ == "__main__":
    import sys
    import os.path
    from future.utils import viewitems

    atomic_data_dir = os.path.normpath('/home/ziletti/nomad/nomad-lab-base/analysis-tools/atomic-data')

    sys.path.insert(0, atomic_data_dir)

    from ai4materials.descriptors.prdf import PRDF
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.wrappers import load_descriptor
    from ase.data import chemical_symbols
    from ai4materials.descriptors.prdf import get_design_matrix
    import scipy.sparse
    import numpy as np
    # read config file
    config_file = '/home/ziletti/Documents/nomadml_docs/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='DEBUG', display_configs=False)

    # import pkg_resources
    #
    # resource_package = 'ai4materials'  # Could be any module/package name
    # # resource_path = '/'.join(('data/nn_models', 'ziletti_et_2018_rgb.json'))  # Do not use os.path.join(), see below
    #
    # resource_path = '/'.join(('data/nn_models', 'ziletti_et_2018_rgb.h5'))  # Do not use os.path.join(), see below
    #
    # filename = pkg_resources.resource_filename('tempfile', "foo.config")
    #
    #
    # template = pkg_resources.resource_string(resource_package, resource_path)
    # # or for a file-like stream:
    # template = pkg_resources.resource_stream(resource_package, resource_path)

    # import pkgutil
    # from ai4materials.utils.utils_config import get_filename
    # filepath = get_filename('ai4materials', 'data/nn_models/ziletti_et_2018_rgb.json')
    #
    # print(filepath)
    #
    #
    # sys.exit(1)

    # setup folder and files
    main_folder = '/home/ziletti/Documents/nomadml_docs/'
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))

    descriptor = PRDF(configs=configs)

    desc_file_name = 'prdf_binaries'
    ase_db_file_binaries = '/home/ziletti/PycharmProjects/ai4materials/ai4materials/data/db_ase/binaries_ghiringhelli2015.db'
    ase_atoms_list = read_ase_db(db_path=ase_db_file_binaries)

    desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                     tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
                                     desc_file=str(desc_file_name)+'.tar.gz', format_geometry='aims',
                                     nb_jobs=-1)

    desc_file_path = '/home/ziletti/Documents/nomadml_docs/desc_folder/prdf_binaries.tar.gz'
    target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    # atom_set = set()
    # for structure in structure_list:
    #     prdfs = structure.info['descriptor']['prdf']
    #
    #     # find the set of unique number of chemical species in the partial radial distribution functions
    #     for (key, value) in viewitems(prdfs):
    #         atom_type_1 = value['particle_atom_number_1']
    #         atom_type_2 = value['particle_atom_number_2']
    #         atom_set.add(atom_type_1)
    #         atom_set.add(atom_type_2)
    #
    #     unique_chem_species = [chemical_symbols[item] for item in atom_set]
    #
    #     max_rdf_length = 0
    #     for name, rdf in viewitems(prdfs):
    #         rdf_length = len(rdf['weights'])
    #         if max_rdf_length < rdf_length:
    #             max_rdf_length = rdf_length
    #
    #         assert len(rdf['arr']) == len(rdf['weights']), "Wrong number of distances at c={}.".format(name)
    #
    # largest_atomic_nb = max(atom_set) + 1
    # logger.debug("Setting up dictionary for cluster calculation.")
    #
    # logger.debug("Longest rdf list: {0}".format(max_rdf_length))
    # logger.debug("Number of different chemical species in the set: {0}".format(len(atom_set)))
    # logger.debug("Highest atomic number in the set: {0}".format(largest_atomic_nb))
    # logger.debug("Actual chemical species set: {0}".format(unique_chem_species))

    total_bins = 50
    max_dist = 25

    X = get_design_matrix(structure_list, total_bins=50, max_dist=25)

    logger.info("Calculation completed.")
