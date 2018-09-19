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

    from ai4materials.descriptors.diffraction3d import Diffraction3D
    from ai4materials.descriptors.diffraction3d import get_design_matrix
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
    from ai4materials.dataprocessing.preprocessing import prepare_dataset
    from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps
    from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
    from ai4materials.dataprocessing.preprocessing import make_data_sets
    from ai4materials.visualization.viewer import Viewer
    from ai4materials.utils.utils_data_retrieval import write_ase_db
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.utils.utils_neural_networks import load_model
    from ai4materials.models.embedding import design_matrix_to_embedding
    from ai4materials.models.clustering import design_matrix_to_clustering
    from ai4materials.wrappers import load_descriptor
    import numpy as np
    from argparse import ArgumentParser
    from functools import partial
    from datetime import datetime
    import numpy as np
    import webbrowser
    import seaborn as sns

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
        db_files_prototypes_basedir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase_prototypes'

    # read config file
    configs = set_configs(main_folder=main_folder)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    dataset_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'datasets')))
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

    descriptor = Diffraction3D(configs=configs)
    # descriptor = Diffraction2D(configs=configs)

    target_nb_atoms = 128
    nb_rotations = 5

    desc_files = [
        'hcp/pristine/A_hP2_194_c_target_nb_atoms128_rotid0_pristine.tar.gz',
        'hcp/pristine/A_hP2_194_c_target_nb_atoms128_rotid1_pristine.tar.gz',
        # 'hcp/pristine/A_hP2_194_c_target_nb_atoms128_rotid2_pristine.tar.gz',
        # 'hcp/pristine/A_hP2_194_c_target_nb_atoms128_rotid3_pristine.tar.gz',
        # 'hcp/pristine/A_hP2_194_c_target_nb_atoms128_rotid4_pristine.tar.gz',
        #
        # 'hcp/vac/A_hP2_194_c_target_nb_atoms128_rotid0_vac05.tar.gz',
        # 'hcp/vac/A_hP2_194_c_target_nb_atoms128_rotid0_vac10.tar.gz',
        # 'hcp/vac/A_hP2_194_c_target_nb_atoms128_rotid0_vac20.tar.gz',
        # 'hcp/vac/A_hP2_194_c_target_nb_atoms128_rotid0_vac30.tar.gz',
        # 'hcp/disp/A_hP2_194_c_target_nb_atoms128_rotid0_disp001.tar.gz',

        'sc/pristine/A_cP1_221_a_target_nb_atoms128_rotid0_pristine.tar.gz',
        'sc/pristine/A_cP1_221_a_target_nb_atoms128_rotid1_pristine.tar.gz',
        # 'sc/pristine/A_cP1_221_a_target_nb_atoms128_rotid2_pristine.tar.gz',
        # 'sc/pristine/A_cP1_221_a_target_nb_atoms128_rotid3_pristine.tar.gz',
        # 'sc/pristine/A_cP1_221_a_target_nb_atoms128_rotid4_pristine.tar.gz',

        'fcc/pristine/A_cF4_225_a_target_nb_atoms128_rotid0_pristine.tar.gz',
        'fcc/pristine/A_cF4_225_a_target_nb_atoms128_rotid1_pristine.tar.gz',
        # 'fcc/pristine/A_cF4_225_a_target_nb_atoms128_rotid2_pristine.tar.gz',
        # 'fcc/pristine/A_cF4_225_a_target_nb_atoms128_rotid3_pristine.tar.gz',
        # 'fcc/pristine/A_cF4_225_a_target_nb_atoms128_rotid4_pristine.tar.gz',

        'diam/pristine/A_cF8_227_a_target_nb_atoms128_rotid0_pristine.tar.gz',
        'diam/pristine/A_cF8_227_a_target_nb_atoms128_rotid1_pristine.tar.gz',
        # 'diam/pristine/A_cF8_227_a_target_nb_atoms128_rotid2_pristine.tar.gz',
        # 'diam/pristine/A_cF8_227_a_target_nb_atoms128_rotid3_pristine.tar.gz',
        # 'diam/pristine/A_cF8_227_a_target_nb_atoms128_rotid4_pristine.tar.gz',

        'bcc/pristine/A_cI2_229_a_target_nb_atoms128_rotid0_pristine.tar.gz',
        'bcc/pristine/A_cI2_229_a_target_nb_atoms128_rotid1_pristine.tar.gz'#,
        # 'bcc/pristine/A_cI2_229_a_target_nb_atoms128_rotid2_pristine.tar.gz',
        # 'bcc/pristine/A_cI2_229_a_target_nb_atoms128_rotid3_pristine.tar.gz',
        # 'bcc/pristine/A_cI2_229_a_target_nb_atoms128_rotid4_pristine.tar.gz'


    ]
    main_desc_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/desc_folder/'

    filename_suffix_pristine = '_pristine.tar.gz'
    # filename_suffix_vac = '_vac25.tar.gz'
    filename_suffix_vac = '_vac20.tar.gz'
    # filename_suffix_disp = '_disp002.tar.gz'
    filename_suffix_disp = '_disp001.tar.gz'

    # desc_files = []
    # for root, dirs, files in os.walk(configs['io']['desc_folder']):
    #     for file_ in files:
    #         if file_.endswith(filename_suffix_pristine) or file_.endswith(filename_suffix_vac) or file_.endswith(filename_suffix_disp):
    #         # if file_.endswith(filename_suffix_vac):
    #             desc_files.append(os.path.join(root, file_))

    # desc_files = ['four_grains/four_grains_poly.xyz_stride_1.0_1.0_20.0_box_size_15.0_pristine.tar.gz']
    # desc_files = ['fcc_crystal_twinning/fcc_crystal_twinning.xyz_stride_0.5_0.5_20.0_box_size_10.0_.tar.gz']

    desc_files = [os.path.join(main_desc_folder, desc_file) for desc_file in desc_files]

    logger.info("Found {} descriptor files".format(len(desc_files)))

    target_list, structure_list = load_descriptor(desc_files=desc_files, configs=configs)

    model_arch_file = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models/enc_dec_1.json'
    model_weights_file = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/saved_models/enc_dec_1.h5'
    nn_model = load_model(model_arch_file, model_weights_file)

    # convolution2d_1
    # embed_params = {'n_jobs': -1, 'n_neighbors': 2}
    # design_matrix = get_design_matrix(structure_list, method='flatten_images')
    # design_matrix = get_design_matrix(structure_list, method='nn_representation', nn_model=nn_model, layer_name='maxpooling2d_1')
    # design_matrix = get_design_matrix(structure_list, method='nn_representation', nn_model=nn_model, layer_name='maxpooling2d_2')
    # design_matrix = get_design_matrix(structure_list, method='nn_representation', nn_model=nn_model, layer_name='flatten_1')
    # design_matrix = get_design_matrix(structure_list, method='nn_representation', nn_model=nn_model, layer_name='dense_1')
    design_matrix = get_design_matrix(structure_list, method='nn_representation', nn_model=nn_model, layer_name='dense_2')

    # umap
    import umap
    # fit = umap.UMAP(n_neighbors=15, n_components=2)
    # fit = umap.UMAP(n_neighbors=100, n_components=2)
    # mapping = fit.fit_transform(design_matrix)

    mapping, embedding = design_matrix_to_embedding(design_matrix, embed_method='pca', embed_params=None)
    # mapping, embedding = design_matrix_to_embedding(design_matrix, embed_method='spect_embed',
    #                                                 embed_params=embed_params)
    x = mapping[:, 0]
    y = mapping[:, 1]

    target = []
    # for structure in structure_list:
    #     target.append('fcc')

    for structure in structure_list:
        target.append(structure.info['spacegroup_0.001'])

    clustering_params = {'n_clusters': 5}
    # clustering_params = None
    # clustering_params = {'eps': 2.0, 'min_samples': 10, 'leaf_size': 1}
    # clust_labels, prob_labels, clustering = design_matrix_to_clustering(design_matrix, clustering_method='kmeans',
    #                                                                    clustering_params=clustering_params)
    # --------------------
    # hdbscan
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    out_hdbscan = clusterer.fit(design_matrix)
    # clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    print(out_hdbscan)
    clust_labels = clusterer.labels_
    # ------------------

    viewer = Viewer(configs=configs)
    # file_html_link, file_html_name = viewer.plot_with_structures(x=x, y=y, target=target,
    #                                                              ase_atoms_list=structure_list,
    #                                                              is_classification=True, target_replicas=(1, 1, 1),
    #                                                              tmp_folder=configs['io']['tmp_folder'])

    # file_html_link, file_html_name = viewer.plot(x=x, y=y, target=target, is_classification=True,
                                                 # tmp_folder=configs['io']['tmp_folder'])

    file_html_link, file_html_name = viewer.plot(x=x, y=y, target=clust_labels, is_classification=True,
                                                 tmp_folder=configs['io']['tmp_folder'])

    print(file_html_name)
    webbrowser.open(file_html_name)

    sys.exit()
