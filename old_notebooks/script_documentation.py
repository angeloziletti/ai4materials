#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016-2018, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/04/18"

if __name__ == "__main__":
    import sys
    import os.path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    common_dir = os.path.normpath(os.path.join(base_dir, "../../../python-common/common/python"))
    nomadml_dir = os.path.normpath(os.path.join(base_dir, "../python-modules/"))
    atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data'))
    apt_dir = os.path.normpath(os.path.join(base_dir, "../../../apt/"))

    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)
        sys.path.insert(0, nomadml_dir)
        sys.path.insert(0, atomic_data_dir)
        sys.path.insert(0, apt_dir)

    from ase.spacegroup import crystal
    from ai4materials.descriptors.diffraction2d import Diffraction2D
    from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import read_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    import numpy as np

    # read config file
    config_file = '/home/ziletti/Documents/nomadml_docs/config_default.yml'
    configs = read_configs(config_file)
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    main_folder = '/home/ziletti/Documents/nomadml_docs'
    desc_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'desc_folder')))
    # checkpoint_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'saved_models')))
    figure_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'attentive_resp_maps')))
    checkpoint_folder = os.path.abspath(os.path.normpath('../assets/data_examples/'))

    # build crystal structures
    fcc_al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=[4.05, 4.05, 4.05, 90, 90, 90])
    bcc_fe = crystal('Fe', [(0, 0, 0)], spacegroup=229, cellpar=[2.87, 2.87, 2.87, 90, 90, 90])
    diamond_c = crystal('C', [(0, 0, 0)], spacegroup=227, cellpar=[3.57, 3.57, 3.57, 90, 90, 90])
    hcp_mg = crystal('Mg', [(1. / 3., 2. / 3., 3. / 4.)], spacegroup=194, cellpar=[3.21, 3.21, 5.21, 90, 90, 120])

    # create supercells - pristine
    fcc_al_supercell = create_supercell(fcc_al, target_nb_atoms=128, cell_type='standard_no_symmetries')
    bcc_fe_supercell = create_supercell(bcc_fe, target_nb_atoms=128, cell_type='standard_no_symmetries')
    diamond_c_supercell = create_supercell(diamond_c, target_nb_atoms=128, cell_type='standard_no_symmetries')
    hcp_mg_supercell = create_supercell(hcp_mg, target_nb_atoms=128, cell_type='standard_no_symmetries')

    ase_atoms_list = [fcc_al_supercell, bcc_fe_supercell, diamond_c_supercell, hcp_mg_supercell]

    # calculate the two-dimensional diffraction fingerprint for all four structures
    descriptor = Diffraction2D(configs=configs)
    diffraction_fingerprints_rgb = [descriptor.calculate(ase_atoms).info['descriptor']['diffraction_2d_intensity'] for ase_atoms in ase_atoms_list]

    neural_network_name = 'ziletti_et_2018_rgb'
    model_weights_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_folder, neural_network_name + '.h5')))
    model_arch_file = os.path.abspath(os.path.normpath(os.path.join(checkpoint_folder, neural_network_name + '.json')))

    # convert list of diffraction fingerprint images to to numpy array
    # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
    images = np.asarray(diffraction_fingerprints_rgb)

    plot_att_response_maps(images, model_arch_file, model_weights_file, figure_folder, nb_conv_layers=6, nb_top_feat_maps=4,
                           layer_nb='all', plot_all_filters=False, plot_filter_sum=True, plot_summary=True)

    logger.info("Calculation completed.")

