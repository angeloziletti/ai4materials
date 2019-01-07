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

import logging
import matplotlib.pyplot as plt
import matplotlib.cm
import os
import numpy as np
from ai4materials.utils.utils_crystals import get_boxes_from_xyz
from ai4materials.wrappers import calc_descriptor
from ai4materials.wrappers import load_descriptor
from ai4materials.dataprocessing.preprocessing import prepare_dataset
from ai4materials.dataprocessing.preprocessing import load_dataset_from_file
from ai4materials.models.cnn_polycrystals import predict
from keras.models import load_model
import pandas as pd
import six.moves.cPickle as pickle

logger = logging.getLogger('ai4materials')


def make_strided_pattern_matching_dataset(polycrystal_file, descriptor, desc_metadata, configs,
                                          operations_on_structure=None, stride_size=(4., 4., 4.), box_size=12.0,
                                          init_sliding_volume=(14., 14., 14.), desc_file=None, desc_only=False,
                                          show_plot_lengths=True, desc_file_suffix_name='_pristine', nb_jobs=-1,
                                          padding_ratio=None, min_nb_atoms=20):
    if desc_file is None:
        logger.info("Calculating system's representation.")
        desc_file = calc_polycrystal_desc(polycrystal_file, stride_size, box_size, descriptor, configs,
                                          desc_file_suffix_name, operations_on_structure, nb_jobs, show_plot_lengths,
                                          padding_ratio=padding_ratio, init_sliding_volume=init_sliding_volume)
    else:
        logger.info("Using the precomputed user-specified descriptor file.")

    if not desc_only:
        target_list, structure_list = load_descriptor(desc_files=desc_file, configs=configs)

        # create dataset
        polycrystal_name = os.path.basename(polycrystal_file)
        dataset_name = '{0}_stride_{1}_{2}_{3}_box_size_{4}_{5}.tar.gz'.format(polycrystal_name, stride_size[0],
                                                                               stride_size[1], stride_size[2], box_size,
                                                                               desc_file_suffix_name)

        # if total number of atoms less than cutoff, set descriptor to NaN
        for structure in structure_list:
            if structure.get_number_of_atoms() <= min_nb_atoms:
                structure.info['descriptor'][desc_metadata][:] = np.nan

        path_to_x_test, path_to_y_test, path_to_summary_test = prepare_dataset(structure_list=structure_list,
                                                                               target_list=target_list,
                                                                               desc_metadata=desc_metadata,
                                                                               dataset_name=dataset_name,
                                                                               target_name='target',
                                                                               target_categorical=True,
                                                                               input_dims=(52, 32), configs=configs,
                                                                               dataset_folder=configs['io'][
                                                                                   'dataset_folder'],
                                                                               main_folder=configs['io']['main_folder'],
                                                                               desc_folder=configs['io']['desc_folder'],
                                                                               tmp_folder=configs['io']['tmp_folder'])

        # get the number of strides in each directions in order to reshape properly
        strided_pattern_pos = []
        for structure in structure_list:
            strided_pattern_pos.append(structure.info['strided_pattern_positions'])

        strided_pattern_pos = np.asarray(strided_pattern_pos)

        path_to_strided_pattern_pos = os.path.abspath(os.path.normpath(os.path.join(configs['io'][
                                                                                   'dataset_folder'],
                                                                                    '{0}_strided_pattern_pos.pkl'.format(
                                                                                        dataset_name))))

        # write to file
        with open(path_to_strided_pattern_pos, 'wb') as output:
            pickle.dump(strided_pattern_pos, output, pickle.HIGHEST_PROTOCOL)
            logger.info("Writing strided pattern positions to {0}".format(path_to_strided_pattern_pos))

        logger.info("Dataset created at {}".format(configs['io']['dataset_folder']))
        logger.info("Strided pattern positions saved at {}".format(configs['io']['dataset_folder']))

    return path_to_x_test, path_to_y_test, path_to_summary_test, path_to_strided_pattern_pos


def get_classification_map(configs, path_to_x_test, path_to_y_test, path_to_summary_test, path_to_strided_pattern_pos,
                           checkpoint_dir, checkpoint_filename,
                           mc_samples=100, interpolation='none', results_file=None, calc_uncertainty=True,
                           conf_matrix_file=None, train_set_name='hcp-bcc-sc-diam-fcc-pristine',
                           cmap_uncertainty='hot',
                           interpolation_uncertainty='none'):

    path_to_x_train = os.path.join(configs['io']['dataset_folder'], train_set_name + '_x.pkl')
    path_to_y_train = os.path.join(configs['io']['dataset_folder'], train_set_name + '_y.pkl')
    path_to_summary_train = os.path.join(configs['io']['dataset_folder'], train_set_name + '_summary.json')

    x_train, y_train, dataset_info_train = load_dataset_from_file(path_to_x=path_to_x_train, path_to_y=path_to_y_train,
                                                                  path_to_summary=path_to_summary_train)

    x_test, y_test, dataset_info_test = load_dataset_from_file(path_to_x=path_to_x_test, path_to_y=path_to_y_test,
                                                               path_to_summary=path_to_summary_test)

    with open(path_to_strided_pattern_pos, 'rb') as input_spm_pos:
        strided_pattern_pos = pickle.load(input_spm_pos)

    logger.debug('Strided_pattern_positions-shape: {0}'.format(strided_pattern_pos.shape))

    params_cnn = {"nb_classes": dataset_info_train["data"][0]["nb_classes"],
                  "classes": dataset_info_train["data"][0]["classes"], "batch_size": 32, "img_channels": 1}

    text_labels = np.asarray(dataset_info_test["data"][0]["text_labels"])
    numerical_labels = np.asarray(dataset_info_test["data"][0]["numerical_labels"])

    filename_no_ext = os.path.abspath(os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename)))

    model = load_model(filename_no_ext)

    results = predict(x_test, y_test, model=model, configs=configs, nb_classes=5, batch_size=params_cnn["batch_size"],
                      mc_samples=mc_samples, conf_matrix_file=conf_matrix_file, numerical_labels=numerical_labels,
                      text_labels=text_labels, results_file=results_file)

    predictive_mean = results['prob_predictions']
    uncertainty = results['uncertainty']

    class_plot_pos = np.asarray(strided_pattern_pos)
    (z_max, y_max, x_max) = np.amax(class_plot_pos, axis=0) + 1

    # make a dataframe to order the prob_predictions
    # this is needed when we read from file - the structures are ordered in a different way after they are saved
    # this comes into play only if more than 10 values for each directions are used
    df_positions = pd.DataFrame(data=class_plot_pos,
                                columns=['strided_pattern_positions_z', 'strided_pattern_positions_y',
                                         'strided_pattern_positions_x'])

    # sort predictive mean
    df_predictive_mean = pd.DataFrame(data=predictive_mean)
    df = pd.concat([df_positions, df_predictive_mean], axis=1, join_axes=[df_positions.index])
    df_predictive_mean_sorted = df.sort_values(
        ['strided_pattern_positions_z', 'strided_pattern_positions_y', 'strided_pattern_positions_x'], ascending=True)

    predictive_mean_sorted = df_predictive_mean_sorted.drop(
        columns=['strided_pattern_positions_z', 'strided_pattern_positions_y', 'strided_pattern_positions_x']).values

    for idx_class in range(predictive_mean_sorted.shape[1]):

        if z_max == 1:
            prob_prediction_class = predictive_mean_sorted[:, idx_class].reshape(y_max, x_max)
        else:
            prob_prediction_class = predictive_mean_sorted[:, idx_class].reshape(z_max, y_max, x_max)

        plot_prediction_heatmaps(prob_prediction_class, title='Probability', class_name=str(idx_class), prefix='prob',
                                 main_folder=configs['io']['main_folder'], cmap='viridis', color_nan='lightgrey',
                                 interpolation=interpolation)

    if calc_uncertainty:
        df_uncertainty = pd.DataFrame()
        for key in uncertainty.keys():
            df_uncertainty[key] = uncertainty[key]

        df = pd.concat([df_positions, df_uncertainty], axis=1, join_axes=[df_positions.index])
        df_uncertainty_sorted = df.sort_values(
            ['strided_pattern_positions_z', 'strided_pattern_positions_y', 'strided_pattern_positions_x'],
            ascending=True)

        uncertainty_sorted = df_uncertainty_sorted.drop(
            columns=['strided_pattern_positions_z', 'strided_pattern_positions_y', 'strided_pattern_positions_x'])

        for key in uncertainty.keys():
            if z_max == 1:
                # make two-dimensional plot
                uncertainty_prediction = uncertainty_sorted[key].values.reshape(y_max, x_max)
            else:
                uncertainty_prediction = uncertainty_sorted[key].values.reshape(z_max, y_max, x_max)

            # for idx_uncertainty in range(predictive_mean_sorted.shape[1]):
            plot_prediction_heatmaps(uncertainty_prediction, title='Uncertainty ({})'.format(str(key)),
                                     main_folder=configs['io']['main_folder'], cmap=cmap_uncertainty,
                                     color_nan='lightgrey',
                                     prefix='uncertainty', suffix=str(key), interpolation=interpolation_uncertainty)


def get_structures_by_boxes(xyz_filename, stride_size, box_size, show_plot_lengths=False, padding_ratio=(1.0, 1.0, 1.0),
                            init_sliding_volume=None):
    xyz_filename_without_ext, xyz_filename_extension = os.path.splitext(os.path.basename(xyz_filename))

    if box_size is not None:
        sliding_volume = [box_size, box_size, box_size]
    else:
        logger.info("Determining box_size automatically.")
        sliding_volume = get_optimal_box_size(xyz_filename, padding_ratio, target_nb_atoms=128, cutoff_percentile=20,
                                              init_sliding_volume=init_sliding_volume)

        assert sliding_volume[0] == sliding_volume[1]
        assert sliding_volume[1] == sliding_volume[2]
        box_size = sliding_volume[0]

    xyz_boxes = get_boxes_from_xyz(xyz_filename, sliding_volume, stride_size, padding_ratio=padding_ratio)
    tot_nb_boxes = len(xyz_boxes) * len(xyz_boxes[0]) * len(xyz_boxes[0][0])

    logger.info("Box size: {}".format(box_size))
    logger.info("Stride size: {}".format(stride_size))
    logger.info(
        "Numbers of boxes in x, y, z: {0} {1} {2}".format(len(xyz_boxes[0][0]), len(xyz_boxes[0]), len(xyz_boxes)))
    logger.info("Total numbers of boxes: {}".format(tot_nb_boxes))

    array_lengths = np.empty_like(xyz_boxes)

    ase_atoms_list = []
    for k in range(len(xyz_boxes)):
        for i in range(len(xyz_boxes[0])):
            for j in range(len(xyz_boxes[0][0])):
                ase_atoms = xyz_boxes[k][i][j]
                array_lengths[k, i, j] = len(ase_atoms)
                # add cell and label
                ase_atoms.set_cell(np.array((box_size, box_size, box_size)) * np.identity(3))
                ase_atoms.set_cell(np.array((box_size, box_size, box_size)) * np.identity(3))
                ase_atoms.info['label'] = xyz_filename_without_ext + '_' + str(k) + '_' + str(i) + '_' + str(j)
                ase_atoms.info['strided_pattern_positions'] = np.asarray((k, i, j))

                ase_atoms_list.append(ase_atoms)

    if show_plot_lengths:
        for idx_slice in range(array_lengths.shape[0]):
            array_lengths_slice = array_lengths[idx_slice].astype(float)
            fig, ax = plt.subplots()
            cax = ax.imshow(array_lengths_slice, interpolation='nearest', cmap=plt.cm.afmhot, origin='lower')
            ax.set_title('Number of atoms in each box for slice {}'.format(idx_slice))
            fig.colorbar(cax)
            plt.show()

    return ase_atoms_list


def get_lengths_from_xyz_boxes(xyz_boxes):
    """From a list of boxes obtained from ``get_structures_by_boxes``, get the number of atoms in each box."""
    array_lengths = np.empty_like(xyz_boxes)

    for k in range(len(xyz_boxes)):
        for i in range(len(xyz_boxes[0])):
            for j in range(len(xyz_boxes[0][0])):
                ase_atoms = xyz_boxes[k][i][j]
                array_lengths[k, i, j] = len(ase_atoms)

    return array_lengths


def get_optimal_box_size(xyz_filename, padding_ratio, target_nb_atoms=128, up_tolerance=80, down_tolerance=30,
                         max_iter=100, init_sliding_volume=None, step=0.1, cutoff_percentile=20):
    # initial guess
    sliding_volume = init_sliding_volume

    logger.info("Target nb_atoms: {}".format(target_nb_atoms))

    for idx in range(max_iter):
        logger.info("Iteration: {}".format(idx))

        xyz_boxes = get_boxes_from_xyz(xyz_filename, sliding_volume, stride_size=(6., 6., 20.),
                                       padding_ratio=padding_ratio)

        lengths = get_lengths_from_xyz_boxes(xyz_boxes).flatten()
        cutoff_nb_atoms = np.percentile(lengths, cutoff_percentile)

        # select all nearest neighbor distances larger than cutoff_nb_atoms
        threshold_indices = np.array(lengths) > cutoff_nb_atoms
        lengths = np.extract(threshold_indices, lengths)

        nb_atoms = np.percentile(lengths, 50)

        logger.info("Mode of the nb_atoms distribution {}".format(nb_atoms))
        logger.info("Sliding box {}".format(sliding_volume))
        logger.info("Diff {}".format(nb_atoms - target_nb_atoms))

        if nb_atoms - up_tolerance <= target_nb_atoms <= nb_atoms + down_tolerance:
            logger.info("Mode of the nb_atoms distribution {}".format(nb_atoms))
            logger.info("nb_atoms - up_tolerance {}".format(nb_atoms - up_tolerance))
            logger.info("nb_atoms + down_tolerance {}".format(nb_atoms + down_tolerance))
            logger.info("Chosen sliding volume: {}".format(sliding_volume))
            break
        elif nb_atoms - target_nb_atoms <= 0:
            logger.debug("Increasing sliding volume: {}".format(sliding_volume))
            sliding_volume = [item + step for item in sliding_volume]
        else:
            logger.debug("Decreasing sliding volume: {}".format(sliding_volume))
            sliding_volume = [item - step for item in sliding_volume]

        del xyz_boxes

    return sliding_volume


def calc_polycrystal_desc(polycrystal_file, stride_size, box_size, descriptor, configs, desc_file_suffix_name,
                          operations_on_structure=None, nb_jobs=-1, show_plot_lengths=True, padding_ratio=None,
                          init_sliding_volume=None):
    ase_atoms_list = get_structures_by_boxes(polycrystal_file, stride_size=stride_size, box_size=box_size,
                                             show_plot_lengths=show_plot_lengths, padding_ratio=padding_ratio,
                                             init_sliding_volume=init_sliding_volume)

    desc_file = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                desc_file='{0}_stride_{1}_{2}_{3}_box_size_{4}_{5}.tar.gz'.format(
                                    os.path.basename(polycrystal_file), stride_size[0], stride_size[1], stride_size[2],
                                    box_size, desc_file_suffix_name), format_geometry='aims',
                                operations_on_structure=operations_on_structure, nb_jobs=nb_jobs)
    return desc_file


def plot_prediction_heatmaps(prob_prediction_class, title, main_folder, class_name='', prefix='prob', suffix='',
                             cmap='viridis', color_nan='black', interpolation='none', vmin=None, vmax=None):
    """

    For available interpolation methods see:
    https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html

    """

    if len(prob_prediction_class.shape) == 2:
        logger.info("Creating two-dimensional plot.")
        fig, ax = plt.subplots()

        cmap = matplotlib.cm.get_cmap(name=cmap)
        # set the color for NaN values
        cmap.set_bad(color=color_nan)

        cax = ax.imshow(prob_prediction_class, interpolation=interpolation, vmin=vmin, vmax=vmax, cmap=cmap,
                        origin='lower')

        if class_name != '':
            ax.set_title('{} for class {}'.format(title, class_name))
        else:
            ax.set_title('{}'.format(title))

        # plt.xlabel(u'x  [\u212B]')
        # plt.ylabel(u'y [\u212B]')
        plt.xlabel(u'x stride #')
        plt.ylabel(u'y stride #')

        # add colorbar, make sure to specify tick locations to match desired ticklabels
        fig.colorbar(cax)
    else:

        # from mayavi import mlab

        azimuth = 0.0
        elevation = 0.0
        # mlab.options.offscreen = False
        # mlab.clf()

        # obj = mlab.contour3d(prob_prediction_class, contours=100, vmin=0.2, vmax=1., opacity=0.)

        filename_npy = os.path.join(main_folder,
                                    '{0}_{1}_class{2}.npy'.format(str(title), str(prefix), str(class_name)))

        np.save(filename_npy, prob_prediction_class)
        logger.info("Voxel array containing the classification map saved at {}.".format(filename_npy))
        # obj.scene.disable_render = True  # obj.scene.anti_aliasing_frames = 0  # mlab.view(azimuth=azimuth, elevation=elevation)  # mlab.colorbar(title='Field intensity', orientation='vertical')  # mlab.show()  #  # mlab.savefig(filename=os.path.join(main_folder, '{0}_class{1}.png'.format(str(prefix),  #                                                                           str(class_name))))  # mlab.close(all=True)

        # logger.info("Creating three-dimensional plot.")  # x = np.arange(prob_prediction_class.shape[0])[:, None, None]  # y = np.arange(prob_prediction_class.shape[1])[None, :, None]  # z = np.arange(prob_prediction_class.shape[2])[None, None, :]  # x, y, z = np.broadcast_arrays(x, y, z)  #  # from mpl_toolkits.mplot3d import Axes3D  # # ax = fig.add_subplot(111, projection='3d')  #  # colmap = cm.ScalarMappable(cmap=plt.cm.Blues)  # colmap.set_array(prob_prediction_class.ravel())  #  # fig = plt.figure(figsize=(8, 6))  # ax = fig.gca(projection='3d')  # ax.scatter(x, y, z, marker='s', s=140, c=prob_prediction_class.ravel(), cmap=plt.cm.Blues, vmin=0, vmax=1,  #            alpha=0.7)  # alpha is transparancey value, 0 (transparent) and 1 (opaque)  # cb = fig.colorbar(colmap)  #  # ax.set_xlabel('x $[\mathrm{\AA}]$')  # ax.set_ylabel('y $[\mathrm{\AA}]$')  # ax.set_zlabel('z $[\mathrm{\AA}]$')  # plt.title(filename)  # plt.show()  # plt.close()  # pl.dump(fig,file(filename+'.pickle','w'))

    if class_name != '':
        filename = os.path.join(main_folder, '{0}_class{1}.pdf'.format(str(prefix), str(class_name)))
        plt.savefig(filename, format='pdf',
                    dpi=1000)
    else:
        filename = os.path.join(main_folder, '{0}_{1}.pdf'.format(str(prefix), str(suffix)))
        plt.savefig(filename, format='pdf', dpi=1000)

    logger.info("File saved to {}.".format(filename))
    plt.close()


def plot_3d_structure():
    from mayavi import mlab
    import numpy as np
    from scipy import ndimage

    filenames = []
    # filenames.append('prob_class0.npy')
    # filenames.append('prob_class1.npy')
    filenames.append('prob_class2.npy')
    # filenames.append('prob_class3.npy')
    filenames.append('prob_class4.npy')
    # filenames.append('uncertainty_class.npy')

    filenames = ['/home/ziletti/Documents/calc_nomadml/rot_inv_3d/' + item for item in filenames]

    for filename in filenames:
        prob = np.load(filename)

        prob = np.nan_to_num(prob)
        prob = np.abs(ndimage.zoom(prob, (6, 6, 6)))
        s = (prob - prob.min()) / (prob.max() - prob.min())

        min = s.min()
        max = s.max()

        mlab.options.offscreen = False
        mlab.clf()
        src = mlab.pipeline.scalar_field(s)

        mlab.pipeline.volume(src, vmin=0., vmax=min + .5 * (max - min))
        mlab.colorbar(title='Field intensity', orientation='vertical')
        # mlab.pipeline.iso_surface(src, contours=[s.min() + 0.1 * s.ptp(), ], opacity=0.1)
        # mlab.pipeline.iso_surface(src, contours=[s.max() - 0.1 * s.ptp(), ], )
        # mlab.volume_slice(s, plane_orientation='x_axes', slice_index=10)
        # mlab.volume_slice(s, plane_orientation='y_axes', slice_index=10)
        # mlab.volume_slice(s, plane_orientation='z_axes', slice_index=10)
        # obj = mlab.contour3d(s, contours=10, vmin=0.2, vmax=1., opacity=0.5)
        # obj.scene.disable_render = True
        # obj.scene.anti_aliasing_frames = 0

        mlab.outline()
        mlab.show()

    mlab.close(all=True)
