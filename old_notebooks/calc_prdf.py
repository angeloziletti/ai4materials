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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data'))
    apt_dir = os.path.normpath(os.path.join(base_dir, "../../../apt/"))

    sys.path.insert(0, atomic_data_dir)
    sys.path.insert(0, apt_dir)

    from ase.spacegroup import crystal
    from ai4materials.descriptors.diffraction2d import Diffraction2D
    from ai4materials.descriptors.prdf import PRDF
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.utils.utils_data_retrieval import generate_facets_input
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.wrappers import load_descriptor

    # read config file
    config_file = '/home/ziletti/Documents/nomadml_docs/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    main_folder = '/home/ziletti/Documents/nomadml_docs/'
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(desc_folder, 'desc_info.json.info')))

    # descriptor = Diffraction2D(configs=configs)
    descriptor = PRDF(configs=configs, rdf_only=True)

    # define operations on structures
    operations_on_structure_list = [(create_supercell,
                                     dict(create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
                                          random_rotation=False, random_rotation_before=True,
                                          cell_type='standard_no_symmetries', optimal_supercell=False)), (
                                    create_vacancies,
                                    dict(target_vacancy_ratio=0.10, create_replicas_by='nb_atoms', min_nb_atoms=32,
                                         target_nb_atoms=256, random_rotation=False, random_rotation_before=True,
                                         cell_type='standard_no_symmetries', optimal_supercell=False)), (
                                    random_displace_atoms, dict(noise_distribution='gaussian', displacement=0.10,
                                                                create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                target_nb_atoms=256, random_rotation=False,
                                                                random_rotation_before=True,
                                                                cell_type='standard_no_symmetries',
                                                                optimal_supercell=False))]

    # =============================================================================
    # Descriptor calculation
    # =============================================================================

    # create the fcc aluminium structure
    nacl = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225, cellpar=[5.64, 5.64, 5.64, 90, 90, 90])

    structure = create_supercell(nacl, target_nb_atoms=256)

    # calculate the two-dimensional diffraction fingerprint
    structure_result = descriptor.calculate(structure)
    rdf = structure_result.info['descriptor']['rdf']

    logger.info("Calculation completed.")
