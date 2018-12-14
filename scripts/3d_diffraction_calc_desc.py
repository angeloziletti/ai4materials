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

    from ase.spacegroup import crystal
    from ai4materials.descriptors.diffraction3d import DISH
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.wrappers import calc_descriptor_in_memory
    from argparse import ArgumentParser
    from datetime import datetime


    startTime = datetime.now()
    now = datetime.now()

    parser = ArgumentParser()
    parser.add_argument("-m", "--machine", dest="machine", help="on which machine the script is run", metavar="MACHINE")
    args = parser.parse_args()

    machine = vars(args)['machine']
    # machine = 'eos'

    if machine == 'eos':
        config_file = '/scratch/ziang/diff_3d/config_prototypes.yml'
        main_folder = '/scratch/ziang/diff_3d/'
        prototypes_basedir = '/scratch/ziang/diff_3d/prototypes_aflow_new/'
        db_files_prototypes_basedir = '/scratch/ziang/diff_3d/db_ase_prototypes'

    else:
        config_file = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/config_diff3d.yml'
        main_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/'
        prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/prototypes_aflow_new'
        db_files_prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase/'

    # read config file
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    checkpoint_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    figure_dir = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'confusion_matrix.png')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    lookup_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'lookup.dat')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'control.json')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))
    filtered_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'filtered_file.json')))
    training_log_file = os.path.abspath(
        os.path.normpath(os.path.join(checkpoint_dir, 'training_' + str(now.isoformat()) + '.log')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'results.csv')))

    configs['io']['dataset_folder'] = dataset_folder
    configs['io']['desc_folder'] = desc_folder

    descriptor = DISH(configs=configs)
    # descriptor = Diffraction2D(configs=configs)

    target_nb_atoms = 128
    nb_rotations = 5

    # define operations on structures
    operations_on_structure_list = [(create_supercell, dict(create_replicas_by='nb_atoms', min_nb_atoms=128,
                                                            target_nb_atoms=target_nb_atoms, random_rotation=True,
                                                            random_rotation_before=True,
                                                            cell_type='standard_no_symmetries',
                                                            optimal_supercell=True)),
                                    (create_vacancies,
                                    dict(target_vacancy_ratio=0.50, create_replicas_by='nb_atoms', min_nb_atoms=32,
                                         target_nb_atoms=128, random_rotation=True, random_rotation_before=True,
                                         cell_type='standard_no_symmetries', optimal_supercell=True)), (
                                    random_displace_atoms,
                                    dict(noise_distribution='uniform_scaled', displacement_scaled=0.005,
                                         create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=128,
                                         random_rotation=True, random_rotation_before=True,
                                         cell_type='standard_no_symmetries', optimal_supercell=True)), (
                                        random_displace_atoms,
                                        dict(noise_distribution='uniform_scaled', displacement_scaled=0.01,
                                             create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=128,
                                             random_rotation=True, random_rotation_before=True,
                                             cell_type='standard_no_symmetries', optimal_supercell=True)), (
                                        random_displace_atoms,
                                        dict(noise_distribution='uniform_scaled', displacement_scaled=0.02,
                                             create_replicas_by='nb_atoms', min_nb_atoms=32, target_nb_atoms=128,
                                             random_rotation=True, random_rotation_before=True,
                                             cell_type='standard_no_symmetries', optimal_supercell=True))]

    # =============================================================================
    # Read prototype data from files
    # =============================================================================
    proto_names = ['A_hP2_194_c', 'A_cP1_221_a', 'A_cF4_225_a', 'A_cF8_227_a', 'A_cI2_229_a']
    # proto_names = ['A_cP1_221_a', 'A_cF4_225_a', 'A_cF8_227_a', 'A_cI2_229_a']
    # proto_names = ['A_cI2_229_a']
    # proto_names = ['A_cF8_227_a']
    # proto_names = ['A_cP1_221_a']

    data_folders = [os.path.join(prototypes_basedir, proto_name) for proto_name in proto_names]
    ase_db_files = [os.path.join(db_files_prototypes_basedir, proto_name) + '.db' for proto_name in proto_names]

    db_protos = zip(proto_names, ase_db_files)
    # for idx_proto, data_folder in enumerate(data_folders):
    #    ase_atoms = read_atomic_structures(data_folder, filename_suffix='.aims', format_input='aims', calc_spgroup=True,
    #                                        symprec=(1e-03, 1e-06, 1e-09))
    #    print(data_folder, len(ase_atoms))

    # add label to structures
    #    for idx_structure, structure in enumerate(ase_atoms):
    #         structure.info['label'] = proto_names[idx_proto] + '_' + str(idx_structure)

    # write an ase db to file for each spacegroup
    #    ase_db_file = write_ase_db(ase_atoms, main_folder, db_name=proto_names[idx_proto], overwrite=True,
    #                                folder_name='db_ase_prototypes')

    #    ase_db_files.append(ase_db_file)

    # for ase_db_file in ase_db_files:
    #    ase_atoms_list = read_ase_db(db_path=ase_db_file)
    #    print(ase_db_file, len(ase_atoms_list))

    # =============================================================================
    # Descriptor calculation
    # =============================================================================

    # for idx_db, db_proto in enumerate(db_protos):
    #     ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])[:6]
    #
    #     print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))
    #
    #     for idx_rot in range(nb_rotations):
    #         desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs,
    #                                                    ase_atoms_list=ase_atoms_list,
    #                                                    tmp_folder=configs['io']['tmp_folder'],
    #                                                    desc_folder=configs['io']['desc_folder'],
    #                                                    # desc_file='try1.tar.gz',
    #                                                    desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp05.tar.gz'.format(
    #                                                        db_proto[0], target_nb_atoms, idx_rot),
    #                                                    # desc_file='{0}_try.tar.gz'.format(
    #                                                    #     db_proto[0]),
    #                                                    format_geometry='aims',
    #                                                    operations_on_structure=operations_on_structure_list[2],
    #                                                    nb_jobs=-1)  # operations_on_structure=None, nb_jobs=1)
    #
    #     for idx_db, db_proto in enumerate(db_protos):
    #         ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])[:6]
    #
    #         print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))
    #
    #         for idx_rot in range(nb_rotations):
    #             desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs,
    #                                                        ase_atoms_list=ase_atoms_list,
    #                                                        tmp_folder=configs['io']['tmp_folder'],
    #                                                        desc_folder=configs['io']['desc_folder'],
    #                                                        # desc_file='try1.tar.gz',
    #                                                        desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp1.tar.gz'.format(
    #                                                            db_proto[0], target_nb_atoms, idx_rot),
    #                                                        # desc_file='{0}_try.tar.gz'.format(
    #                                                        #     db_proto[0]),
    #                                                        format_geometry='aims',
    #                                                        operations_on_structure=operations_on_structure_list[3],
    #                                                        nb_jobs=-1)  # operations_on_structure=None, nb_jobs=1)
    #
    #     for idx_db, db_proto in enumerate(db_protos):
    #         ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])[:6]
    #
    #         print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))
    #
    #         for idx_rot in range(nb_rotations):
    #             desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs,
    #                                                        ase_atoms_list=ase_atoms_list,
    #                                                        tmp_folder=configs['io']['tmp_folder'],
    #                                                        desc_folder=configs['io']['desc_folder'],
    #                                                        # desc_file='try1.tar.gz',
    #                                                        desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp2.tar.gz'.format(
    #                                                            db_proto[0], target_nb_atoms, idx_rot),
    #                                                        # desc_file='{0}_try.tar.gz'.format(
    #                                                        #     db_proto[0]),
    #                                                        format_geometry='aims',
    #                                                        operations_on_structure=operations_on_structure_list[4],
    #                                                        nb_jobs=-1)  # operations_on_structure=None, nb_jobs=1)

    # build atomic structure

    ase_atoms_list = []
    structure = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                        cellpar=[5., 5., 5., 90, 90, 90])

    structure_2 = crystal(['Na', 'Cl'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
                        cellpar=[5., 5., 5., 90, 90, 90])

    ase_atoms_list.append(structure)
    ase_atoms_list.append(structure_2)

    desc_file_path = calc_descriptor_in_memory(descriptor=descriptor, configs=configs,
                                               ase_atoms_list=ase_atoms_list,
                                               tmp_folder=configs['io']['tmp_folder'],
                                               desc_folder=configs['io']['desc_folder'],
                                               desc_file='try1.tar.gz',
                                               format_geometry='aims',
                                               operations_on_structure=operations_on_structure_list[0],
                                               nb_jobs=-1)  # operations_on_structure=None, nb_jobs=1)