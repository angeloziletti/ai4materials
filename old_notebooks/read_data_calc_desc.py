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

    atomic_data_dir = os.path.normpath('/home/ziletti/nomad/nomad-lab-base/analysis-tools/atomic-data')

    sys.path.insert(0, atomic_data_dir)

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
    descriptor = PRDF(configs=configs)

    # define operations on structures
    operations_on_structure_list = [(create_supercell,
                                     dict(create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=256,
                                          random_rotation=False, random_rotation_before=True,
                                          cell_type='standard_no_symmetries', optimal_supercell=False)),
                                    (create_vacancies,
                                     dict(target_vacancy_ratio=0.10,
                                          create_replicas_by='nb_atoms',
                                          min_nb_atoms=32,
                                          target_nb_atoms=256,
                                          random_rotation=False,
                                          random_rotation_before=True,
                                          cell_type='standard_no_symmetries',
                                          optimal_supercell=False)),
                                    (random_displace_atoms,
                                     dict(noise_distribution='gaussian', displacement=0.10,
                                          create_replicas_by='nb_atoms', min_nb_atoms=32,
                                          target_nb_atoms=256, random_rotation=False,
                                          random_rotation_before=True, cell_type='standard_no_symmetries',
                                          optimal_supercell=False))]

    # =============================================================================
    # Descriptor calculation
    # =============================================================================

    # desc_file_name = 'desc_calc_trial'
    desc_file_name = 'try1'
    # ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_1e-9_139.db'
    ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_7_classes.db'
    # ase_db_file = '/home/ziletti/Documents/calc_xray/2d_nature_comm/db_ase/elemental_solids_ncomms_1e-3_1e-6_pristine.db'

    ase_atoms_list = read_ase_db(db_path=ase_db_file)

    desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                     tmp_folder=tmp_folder, desc_folder=desc_folder, desc_info_file=desc_info_file,
                                     desc_file=str(desc_file_name)+'.tar.gz', format_geometry='aims',
                                     operations_on_structure=operations_on_structure_list[1],
                                     nb_jobs=-1)

    desc_file_path = '/home/ziletti/Documents/nomadml_docs/desc_folder/try1.tar.gz'
    target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    ase_db_file = write_ase_db(ase_atoms_list=structure_list,
                               db_name='elemental_solids_ncomms_7_classes_new', main_folder=main_folder,
                               folder_name='db_ase')

    desc_file_path = '/home/ziletti/Documents/nomadml_docs/desc_folder/try1.tar.gz'
    target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)

    sys.exit()

    df, sprite_atlas = generate_facets_input(structure_list=structure_list, desc_metadata='diffraction_2d_intensity',
                                             target_list=target_list,
                                             sprite_atlas_filename=desc_file_name,
                                             configs=configs, normalize=True)

    logger.info("Calculation completed.")