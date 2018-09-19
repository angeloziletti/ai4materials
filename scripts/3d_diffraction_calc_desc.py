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

    from ase.spacegroup import get_spacegroup
    from ai4materials.descriptors.diffraction3d import DISH
    from ai4materials.descriptors.diffraction2d import Diffraction2D
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    from ai4materials.utils.utils_crystals import create_vacancies
    from ai4materials.utils.utils_crystals import random_displace_atoms
    from ai4materials.visualization.viewer import Viewer
    import matplotlib.cm as cm
    from ai4materials.utils.utils_data_retrieval import clean_folder
    from ai4materials.utils.utils_data_retrieval import generate_facets_input
    from ai4materials.utils.utils_parsing import read_atomic_structures
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import make_data_sets
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.visualization.viewer import Viewer
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.wrappers import load_descriptor
    import numpy as np
    from ai4materials.models.cnn_architectures import model_cnn_rot_inv
    from argparse import ArgumentParser
    from functools import partial
    from datetime import datetime
    import numpy as np
    import webbrowser

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
    nb_rotations = 1

    # define operations on structures
    operations_on_structure_list = [(create_supercell, dict(create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                            target_nb_atoms=target_nb_atoms, random_rotation=True,
                                                            random_rotation_before=True,
                                                            cell_type='standard_no_symmetries',
                                                            optimal_supercell=True)), (create_vacancies,
                                                                                       dict(target_vacancy_ratio=0.01,
                                                                                            create_replicas_by='nb_atoms',
                                                                                            min_nb_atoms=32,
                                                                                            target_nb_atoms=128,
                                                                                            random_rotation=True,
                                                                                            random_rotation_before=True,
                                                                                            cell_type='standard_no_symmetries',
                                                                                            optimal_supercell=True)),
(create_vacancies,
                                                                                       dict(target_vacancy_ratio=0.02,
                                                                                            create_replicas_by='nb_atoms',
                                                                                            min_nb_atoms=32,
                                                                                            target_nb_atoms=128,
                                                                                            random_rotation=True,
                                                                                            random_rotation_before=True,
                                                                                            cell_type='standard_no_symmetries',
                                                                                            optimal_supercell=True)),
(create_vacancies,
                                                                                       dict(target_vacancy_ratio=0.05,
                                                                                            create_replicas_by='nb_atoms',
                                                                                            min_nb_atoms=32,
                                                                                            target_nb_atoms=128,
                                                                                            random_rotation=True,
                                                                                            random_rotation_before=True,
                                                                                            cell_type='standard_no_symmetries',
                                                                                            optimal_supercell=True)), 
					(random_displace_atoms, dict(noise_distribution='uniform_scaled', displacement_scaled=0.008,
                                                                    create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                    target_nb_atoms=128, random_rotation=True,
                                                                    random_rotation_before=True,
                                                                    cell_type='standard_no_symmetries',
                                                                    optimal_supercell=True)),
                                        (random_displace_atoms, dict(noise_distribution='uniform_scaled', displacement_scaled=0.016,
                                                                    create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                    target_nb_atoms=128, random_rotation=True,
                                                                    random_rotation_before=True,
                                                                    cell_type='standard_no_symmetries',
                                                                    optimal_supercell=True)),
                                        (random_displace_atoms, dict(noise_distribution='uniform_scaled', displacement_scaled=0.045,
                                                                    create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                    target_nb_atoms=128, random_rotation=True,
                                                                    random_rotation_before=True,
                                                                    cell_type='standard_no_symmetries',
                                                                    optimal_supercell=True)),
                                       (random_displace_atoms, dict(noise_distribution='uniform_scaled', displacement_scaled=0.050,
                                                                    create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                    target_nb_atoms=128, random_rotation=True,
                                                                    random_rotation_before=True,
                                                                    cell_type='standard_no_symmetries',
                                                                    optimal_supercell=True)),
                                    (random_displace_atoms, dict(noise_distribution='uniform_scaled', displacement_scaled=0.075,
                                                                 create_replicas_by='nb_atoms', min_nb_atoms=32,
                                                                 target_nb_atoms=128, random_rotation=True,
                                                                 random_rotation_before=True, cell_type='standard_no_symmetries',
                                                                 optimal_supercell=True))
                                    ]

    # =============================================================================
    # Read prototype data from files
    # =============================================================================
    proto_names = ['A_hP2_194_c', 'A_cP1_221_a', 'A_cF4_225_a', 'A_cF8_227_a', 'A_cI2_229_a']
    # proto_names = ['A_cP1_221_a', 'A_cF4_225_a', 'A_cF8_227_a', 'A_cI2_229_a']
    # proto_names = ['A_cI2_229_a']
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

    for idx_db, db_proto in enumerate(db_protos):
        ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])[:1]

        print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))

        for idx_rot in range(nb_rotations):

            desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                             tmp_folder=configs['io']['tmp_folder'], desc_folder=configs['io']['desc_folder'],
                                             # desc_file='try1.tar.gz',
                                             # desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp0008.tar.gz'.format(db_proto[0],
                                             #                                                  target_nb_atoms, idx_rot),
                                             desc_file='{0}_try.tar.gz'.format(
                                                 db_proto[0]),
                                             format_geometry='`aims',
                                             operations_on_structure=operations_on_structure_list[0], nb_jobs=1)
                                             # operations_on_structure=None, nb_jobs=1)

        print(desc_file_path)

    sys.exit()

    for idx_db, db_proto in enumerate(db_protos):
        ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])

        print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))

        for idx_rot in range(nb_rotations):

            desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                             tmp_folder=configs['io']['tmp_folder'], desc_folder=configs['io']['desc_folder'],
                                             desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp0016.tar.gz'.format(db_proto[0],
                                                                                              target_nb_atoms, idx_rot),
                                             format_geometry='aims',
                                             operations_on_structure=operations_on_structure_list[5], nb_jobs=6)
        print(desc_file_path)


    for idx_db, db_proto in enumerate(db_protos):
        ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])

        print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))

        for idx_rot in range(nb_rotations):

            desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                             tmp_folder=configs['io']['tmp_folder'], desc_folder=configs['io']['desc_folder'],
                                             desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp0045.tar.gz'.format(db_proto[0],
                                                                                              target_nb_atoms, idx_rot),
                                             format_geometry='aims',
                                             operations_on_structure=operations_on_structure_list[6], nb_jobs=6)
        print(desc_file_path)


    for idx_db, db_proto in enumerate(db_protos):
        ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])

        print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))

        for idx_rot in range(nb_rotations):

            desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                             tmp_folder=configs['io']['tmp_folder'], desc_folder=configs['io']['desc_folder'],
                                             desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp0050.tar.gz'.format(db_proto[0],
                                                                                              target_nb_atoms, idx_rot),
                                             format_geometry='aims',
                                             operations_on_structure=operations_on_structure_list[7], nb_jobs=6)
        print(desc_file_path)

    for idx_db, db_proto in enumerate(db_protos):
        ase_atoms_list = read_ase_db(db_path=ase_db_files[idx_db])

        print('{} structures for prototype {}'.format(len(ase_atoms_list), db_proto[0]))

        for idx_rot in range(nb_rotations):

            desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                             tmp_folder=configs['io']['tmp_folder'], desc_folder=configs['io']['desc_folder'],
                                             desc_file='{0}_target_nb_atoms{1}_rotid{2}_disp0075.tar.gz'.format(db_proto[0],
                                                                                              target_nb_atoms, idx_rot),
                                             format_geometry='aims',
                                             operations_on_structure=operations_on_structure_list[8], nb_jobs=6)
        print(desc_file_path)



    sys.exit()
    # filename_suffix = '_pristine.tar.gz'
    # filename_suffix_vac = '_vac25.tar.gz'
    # filename_suffix_vac = '_vac20.tar.gz'
    # filename_suffix_disp = '_disp002.tar.gz'
    # filename_suffix_disp = '_disp001.tar.gz'

    # #
    # desc_file_paths = []
    # for root, dirs, files in os.walk(configs['io']['desc_folder']):
    #     for file_ in files:
    #         # if file_.endswith(filename_suffix_vac) or file_.endswith(filename_suffix_disp):
    #         if file_.endswith(filename_suffix_vac):
    #             desc_file_paths.append(os.path.join(root, file_))
    #
    # logger.info("Found {} descriptor files".format(len(desc_file_paths)))
    #
    # target_list, structure_list = load_descriptor(desc_files=desc_file_paths, configs=configs)
    # #
    # ase_db_file = write_ase_db(ase_atoms_list=structure_list, main_folder=configs['io']['main_folder'],
    #                            db_name='vac20', db_type='db', overwrite=True,
    #                            folder_name='db_ase')
    #
    # # calculate spacegroup for each structure and put it in target_list
    # spgroup_list = []
    # for idx_str, structure in enumerate(structure_list):
    #     target_list[idx_str]['data'][0]['spacegroup_0.001'] = structure.info['spacegroup_0.001']
    #
    # path_to_x, path_to_y, path_to_summary = prepare_dataset(structure_list=structure_list, target_list=target_list,
    #                                                         desc_metadata='diffraction_3d_sh_spectrum',
    #                                                         dataset_name='hcp-bcc-sc-diam-fcc-disp001',
    #                                                         target_name='spacegroup_0.001',
    #                                                         target_categorical=True,
    #                                                         input_dims=(52, 32), configs=configs,
    #                                                         dataset_folder=dataset_folder, main_folder=main_folder,
    #                                                         desc_folder=configs['io']['desc_folder'],
    #                                                         tmp_folder=configs['io']['tmp_folder'],
    #                                                         notes="Dataset with 5 rotations.")

    train_set_name = 'hcp-bcc-sc-diam-fcc-pristine'
    path_to_x_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')))
    path_to_y_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')))
    path_to_summary_train = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')))

    test_set_name = 'hcp-bcc-sc-diam-fcc-vac25'
    # test_set_name = 'hcp-bcc-sc-diam-fcc-disp002'
    # test_set_name = 'hcp-bcc-sc-diam-fcc-disp001'
    # test_set_name = 'hcp-bcc-sc-diam-fcc-vac25-disp002'
    path_to_x_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_x.pkl')))
    path_to_y_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_y.pkl')))
    path_to_summary_test = os.path.abspath(
        os.path.normpath(os.path.join(configs['io']['dataset_folder'], test_set_name + '_summary.json')))

    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                               path_to_summary=path_to_summary_test)

    params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
                  "classes": dataset_info_train["data"][0]["classes"],
                  # "checkpoint_filename": 'try_' + str(now.isoformat()),
                  "checkpoint_filename": 'enc_dec_no_batch_norm',
                  # "checkpoint_filename": 'try_2018-08-11T15:39:50.802037',
                  # "checkpoint_filename": 'rot_inv_kernel_15',
                  "batch_size": 32, "img_channels": 1}

    text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

    data_set_train = make_data_sets(x_train_val=x_train, y_train_val=y_train, split_train_val=True, test_size=0.1,
                                    x_test=x_test, y_test=y_test, flatten_images=False)

    # beautiful maps
    #        conv2d_filters=[32, 16, 12, 8, 4, 4],
    #        kernel_sizes=[3, 3, 3, 3, 3, 3],
    # hidden_layer_size = 32)

    # partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[32, 32, 16, 16, 16, 16],
    #                                      kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64)

    partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[32, 16, 8, 8, 16, 32],
                                     kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64, dropout=0.25)

    # partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[8, 8, 8, 8, 8, 8],
    #                                  kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64)

    # partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[8, 8, 8, 8, 8, 8],
    #                                  kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64)


    # best for disp002 - best so far
    # partial_model_architecture = partial(model_cnn_rot_inv, conv2d_filters=[32, 32, 16, 16, 8, 8],
    #                                  kernel_sizes=[3, 3, 3, 3, 3, 3], hidden_layer_size=64)
    #
    # partial_model_architecture = partial(model_fully_conv, conv2d_filters=[32, 32, 16, 16, 8, 512],
    #                                      kernel_sizes=[3, 3, 3, 3, 3])

    data_set_predict = make_data_sets(x_train_val=x_test, y_train_val=y_test, split_train_val=False, test_size=0.1,
                                      x_test=x_test, y_test=y_test)

    # train_cnn_keras(data_set=data_set_train, configs=configs, nb_classes=params_cnn["nb_classes"],
    #                 partial_model_architecture=partial_model_architecture, batch_size=params_cnn["batch_size"],
    #                 img_channels=1, checkpoint_dir=checkpoint_dir, checkpoint_filename=params_cnn["checkpoint_filename"],
    #                 nb_epoch=5, training_log_file=training_log_file, early_stopping=False,
    #                 normalize=True)

    target_pred_class, target_pred_probs, prob_predictions, conf_matrix, uncertainty = predict_cnn_keras(data_set_predict,
                                                                                            params_cnn["nb_classes"],
                                                                                            configs=configs,
                                                                                            batch_size=params_cnn[
                                                                                                "batch_size"],
                                                                                            checkpoint_dir=checkpoint_dir,
                                                                                            checkpoint_filename=
                                                                                            params_cnn[
                                                                                                "checkpoint_filename"],
                                                                                            show_model_acc=True,
                                                                                            predict_probabilities=True,
                                                                                            plot_conf_matrix=True,
                                                                                            conf_matrix_file=conf_matrix_file,
                                                                                            numerical_labels=numerical_labels,
                                                                                            text_labels=text_labels,
                                                                                            results_file=results_file,
                                                                                            normalize=True)


    ase_db_file = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase/pristine.db'

    ase_atoms_list = read_ase_db(ase_db_file)[:10]

    viewer = Viewer(configs=configs)
    n_structs = len(ase_atoms_list)

    x = np.random.rand(n_structs)
    y = np.random.rand(n_structs)
    target = target_pred_class[:10]

    file_html_link, file_html_name = viewer.plot_with_structures(x=x, y=y, target=target,
                                                                 ase_atoms_list=ase_atoms_list,
                                                                 is_classification=True, target_replicas=(1, 1, 1),
                                                                 tmp_folder=configs['io']['tmp_folder'])

    print(file_html_name)
    webbrowser.open(file_html_name)

    sys.exit()

    model_weights_file = os.path.join(checkpoint_dir, '{0}.h5'.format(
        params_cnn['checkpoint_filename']))

    model_arch_file = os.path.join(checkpoint_dir, '{0}.json'.format(
        params_cnn['checkpoint_filename']))

    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    nb_imgs = 2
    images_classes = []
    images_class_0 = x_train[(y_train == 0)]
    images_class_1 = x_train[(y_train == 1)]
    images_class_2 = x_train[(y_train == 2)]
    images_class_3 = x_train[(y_train == 3)]
    images_class_4 = x_train[(y_train == 4)]

    images_classes.append(images_class_0)
    images_classes.append(images_class_1)
    images_classes.append(images_class_2)
    images_classes.append(images_class_3)
    images_classes.append(images_class_4)

    nb_classes = 5
    for idx_cl in range(nb_classes):
        images = images_classes[idx_cl][:nb_imgs, :, :].reshape(nb_imgs, x_train.shape[1], x_train.shape[2], 1)
        plot_att_response_maps(images, model_arch_file=model_arch_file, model_weights_file=model_weights_file,
                               cmap=cm.viridis,
                               figure_dir=figure_dir+'_class'+str(idx_cl), nb_conv_layers=6, nb_top_feat_maps=4, layer_nb='all',
                               plot_all_filters=True, plot_filter_sum=True, plot_summary=True)
