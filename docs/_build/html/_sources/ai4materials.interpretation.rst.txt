Neural network interpretation
===================================

Understanding why a machine learning algorithm arrives at the classification decision is of paramount importance,
especially in the natural sciences. For deep learning models this is particularly challenging because of their tendency to represent information in a highly distributed manner,
and the presence of non-linearities in the network’s layers.

Here we provide a materials science use case of interpretable machine learning for crystal-structure classification from Ziletti et al. (2018) [1]_.

Example: attentive response maps in deep-learning-driven crystal recognition
----------------------------------------------------------------------------
.. module:: ai4materials.interpretation.plot_att_response_maps-example-diffraction2d
   :synopsis: Attentive response maps for diffraction fingerprints


This example shows how to identify the regions in the image that are the most important in the neural network's classification decision.
In particular, attentive response maps are calculated using the fractionally strided convolutional technique
introduced by Zeiler and Fergus (2014) [2]_, and applied for the first time in materials science by Ziletti et al. (2018) [1]_.

The steps performed in the code below are the following:

* define the folders where the results are going to be saved
* build four crystal structures (bcc, fcc, diam, hcp) using the ASE package
* create a pristine supercell using the function :py:mod:`ai4materials.utils.utils_crystals.create_supercell`
* calculate the two-dimensional diffraction fingerprint for all four crystal structures (a RGB image) with from :py:mod:`ai4materials.descriptors.diffraction2d.Diffraction2D`
* obtain the attentive response maps for each diffraction fingerprints with :py:mod:`ai4materials.interpretation.deconv_resp_maps.plot_att_response_maps`.
  These identify the parts of the image that are more important in the classification decision.


.. testcode::

    from ase.spacegroup import crystal
    from ai4materials.descriptors.diffraction2d import Diffraction2D
    from ai4materials.interpretation.deconv_resp_maps import plot_att_response_maps
    from ai4materials.utils.utils_config import get_data_filename
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_crystals import create_supercell
    import numpy as np
    import os.path

    # set configs
    configs = set_configs(main_folder='./nn_interpretation_ai4materials/')
    logger = setup_logger(configs, level='INFO', display_configs=False)

    # setup folder and files
    # checkpoint_folder = os.path.join(configs['io']['main_folder'], 'saved_models')
    figure_folder = os.path.join(configs['io']['main_folder'], 'attentive_resp_maps')

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

    model_weights_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.h5')
    model_arch_file = get_data_filename('data/nn_models/ziletti_et_2018_rgb.json')

    # convert list of diffraction fingerprint images to to numpy array
    # images needs to be a numpy array with shape (n_images, dim1, dim2, channels)
    images = np.asarray(diffraction_fingerprints_rgb)

    plot_att_response_maps(images, model_arch_file, model_weights_file, figure_folder, nb_conv_layers=6, nb_top_feat_maps=4,
                           layer_nb='all', plot_all_filters=False, plot_filter_sum=True, plot_summary=True)


In each image below we show:

* (left) original image to be classified corresponding to the two-dimensional diffraction fingerprint of a given structure
* (center) attentive response maps from the top four most activated filters (red channel) for the diffraction fingerprint. The brighter the pixel, the most important is that location for classification
* (right) sum of the last convolutional layer attentive response maps

for the case of a face-centered-cubic structure:

.. image:: attentive_resp_maps_fcc_red.png

and a body-centered-cubic structure:

.. image:: attentive_resp_maps_bcc_red.png

From the attentive response maps (center), we notice that the convolutional neural network filters are composed in a hierarchical fashion, increasing
their complexity from one layer to another. At the third convolutional layer, the neural network discovers that the diffraction peaks, and their relative
arrangement, are the most effective way to predict crystal classes (as a human expert would do).
Furthermore, from the sum of the last convolutional layer attentive response maps, we observe that the neural network learned crystal templates automatically from the data.


.. [1] A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli, "Insightful classification of crystal structures
   using deep learning," Nature Communications, vol. 9, pp. 2775, 2018.
   [`Link to article <https://www.nature.com/articles/s41467-018-05169-6>`_]

.. [2] D. M. Zeiler, and R. Fergus, "Visualizing and understanding convolutional networks,"
   European Conference on Computer Vision, Springer. pp. 818, 2014.
   [`Link to article <https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf>`_]


.. sectionauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>


Submodules
----------

.. toctree::

   ai4materials.interpretation.deconv_resp_maps

Module contents
---------------

.. automodule:: ai4materials.interpretation
    :members:
    :undoc-members:
    :show-inheritance:
