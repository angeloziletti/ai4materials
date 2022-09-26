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

import itertools

import logging
import pandas as pd
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
os.system("export DISPLAY=:0")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show, axes, sci
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from numpy import amin, amax, ravel
from matplotlib.colors import LinearSegmentedColormap

import tensorflow as tf

# tf.set_random_seed(0) # for tf<1
tf.random.set_seed(0)

logger = logging.getLogger('ai4materials')


def insert_newlines(string, every=64):
    return '\n'.join(string[i:i + every] for i in range(0, len(string), every))


def plot_sph_harmonics():
    # http://docs.enthought.com/mayavi/mayavi/auto/example_spherical_harmonics.html

    from mayavi import mlab
    import numpy as np
    from scipy.special import sph_harm

    # Create a sphere
    r = 0.3
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
    mlab.clf()
    # Represent spherical harmonics on the surface of the sphere
    for n in range(1, 6):
        for m in range(n):
            s = sph_harm(m, n, theta, phi).real

            mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')

            s[s < 0] *= 0.97

            s /= s.max()
            mlab.mesh(s * x - m, s * y - n, s * z + 1.3, scalars=s, colormap='Spectral')

    mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
    mlab.show()


def plot_save_cnn_results(filename, accuracy=True, cross_entropy_loss=True, show_plot=False):
    """Plot and save results of a convolutional neural network calculation
        from the .csv file written by Keras CSVLogger.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    df_results = pd.read_csv(filename)
    plt.style.use('fivethirtyeight')
    epoch = df_results.epoch.values + 1

    if accuracy:
        a_tr = df_results.acc.values * 100.0
        a_val = df_results.val_acc.values * 100.0
    #        a_test = df_results.val_acc.values*100.0

    if cross_entropy_loss:
        c_tr = df_results.loss.values * 100.0
        c_val = df_results.val_loss.values * 100.0
    #        c_test = df_results.val_loss.values*100.0

    if accuracy:
        figure_a = make_plot_accuracy(epoch, a_tr, a_val)
        # save png file (same name as csv file, but with png extension)
        figure_a.savefig(filename.rsplit('.', 1)[0] + '_accuracy.png', format="png")

        if show_plot:
            figure_a.show()
    if cross_entropy_loss:
        figure_c = make_plot_cross_entropy_loss(epoch, c_tr, c_val)
        # save png file (same name as csv file, but with png extension)
        figure_c.savefig(filename.rsplit('.', 1)[0] + '_cross_entropy_loss.png', format="png")

        if show_plot:
            figure_c.show()


def make_plot_accuracy(step, train_data, val_data):
    # add mask to have line between missing values
    train_data_mask = np.isfinite(train_data)
    val_data_mask = np.isfinite(val_data)

    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(13, 10))
    plt.suptitle("Convolutional neural network: model accuracy", fontname='Ubuntu', fontsize=24, fontstyle='italic',
                 fontweight='bold')
    plt.tight_layout(pad=4.0, w_pad=2.0, h_pad=3.0)
    plt.grid(True)
    ax1.set_xlim([-np.amax(step) * 0.01 + 1.0, np.amax(step) * 1.01])
    ax1.set_ylim([0, 105.0])
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(min(step), max(step) + 1, 1))
    ax1.plot(step[train_data_mask], train_data[train_data_mask], 'ro-', label='Training accuracy')
    ax1.plot(step[val_data_mask], val_data[val_data_mask], 'go-', label='Validation accuracy')
    ax1.set_xlabel('Epoch number')
    ax1.set_ylabel('Accuracy [%]')
    ax1.set_axis_bgcolor((224 / 255, 224 / 255, 224 / 255))

    legend = ax1.legend(loc='lower right', borderaxespad=0., frameon=1)
    for text in legend.get_texts():
        plt.setp(text, color=(224 / 255, 224 / 255, 224 / 255))
    frame = legend.get_frame()
    frame.set_facecolor((32 / 255, 32 / 255, 32 / 255))
    frame.set_edgecolor((32 / 255, 32 / 255, 32 / 255))
    ax2.set_xlim([-np.amax(step) * 0.01 + 1.0, np.amax(step) * 1.01])
    ax2.set_ylim([95, 100.5])
    ax2.xaxis.set_ticks(np.arange(min(step), max(step) + 1, 1))
    ax2.plot(step[train_data_mask], train_data[train_data_mask], 'ro-', label='Training accuracy')
    ax2.plot(step[val_data_mask], val_data[val_data_mask], 'go-', label='Validation accuracy')
    ax2.set_xlabel('Epoch number')
    ax2.set_ylabel('Accuracy [%]')
    ax2.set_axis_bgcolor((224 / 255, 224 / 255, 224 / 255))

    legend = ax2.legend(loc='lower right', borderaxespad=0., frameon=1)
    for text in legend.get_texts():
        plt.setp(text, color=(224 / 255, 224 / 255, 224 / 255))
    frame = legend.get_frame()
    frame.set_facecolor((32 / 255, 32 / 255, 32 / 255))
    frame.set_edgecolor((32 / 255, 32 / 255, 32 / 255))
    return plt


def make_plot_cross_entropy_loss(step, train_data, val_data, title=None):
    # add mask to have line between missing values
    train_data_mask = np.isfinite(train_data)
    val_data_mask = np.isfinite(val_data)

    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(13, 10))
    plt.suptitle("Convolutional neural network: cross-entropy loss", fontname='Ubuntu', fontsize=24, fontstyle='italic',
                 fontweight='bold')
    plt.tight_layout(pad=4.0, w_pad=2.0, h_pad=3.0)
    plt.grid(True)
    ax1.set_xlim([-np.amax(step) * 0.01 + 1.0, np.amax(step) * 1.01])
    min_value = min(np.nanmin(train_data), np.nanmin(val_data))
    max_value = max(np.nanmax(train_data), np.nanmax(val_data))
    ax1.set_ylim([-max_value * 0.05, max_value * 1.03])
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(min(step), max(step) + 1, 1))
    ax1.plot(step[train_data_mask], train_data[train_data_mask], 'ro-', label='Training accuracy')
    ax1.plot(step[val_data_mask], val_data[val_data_mask], 'go-', label='Validation accuracy')
    ax1.set_xlabel('Epoch number')
    ax1.set_ylabel('Cross entropy loss')
    ax1.set_axis_bgcolor((224 / 255, 224 / 255, 224 / 255))

    legend = ax1.legend(loc='upper right', borderaxespad=0., frameon=1)
    for text in legend.get_texts():
        plt.setp(text, color=(224 / 255, 224 / 255, 224 / 255))
    frame = legend.get_frame()
    frame.set_facecolor((32 / 255, 32 / 255, 32 / 255))
    frame.set_edgecolor((32 / 255, 32 / 255, 32 / 255))
    ax2.set_xlim([-np.amax(step) * 0.01 + 1.0, np.amax(step) * 1.01])
    min_value = min(np.nanmin(train_data), np.nanmin(val_data))
    max_value = max(np.nanmax(train_data), np.nanmax(val_data))
    ax2.set_ylim([-1.0 + min_value * 0.97, max_value * 0.20])
    ax2.xaxis.set_ticks(np.arange(min(step), max(step) + 1, 1))
    ax2.plot(step[train_data_mask], train_data[train_data_mask], 'ro-', label='Training accuracy')
    ax2.plot(step[val_data_mask], val_data[val_data_mask], 'go-', label='Validation accuracy')
    ax2.set_xlabel('Epoch number')
    ax2.set_ylabel('Cross entropy loss')
    ax2.set_axis_bgcolor((224 / 255, 224 / 255, 224 / 255))

    legend = ax2.legend(loc='upper right', borderaxespad=0., frameon=1)
    for text in legend.get_texts():
        plt.setp(text, color=(224 / 255, 224 / 255, 224 / 255))
    frame = legend.get_frame()
    frame.set_facecolor((32 / 255, 32 / 255, 32 / 255))
    frame.set_edgecolor((32 / 255, 32 / 255, 32 / 255))
    return plt


def aggregate_struct_trans_data(filename, nb_rows_to_cut=0, nb_samples=None, nb_order_param_steps=None,
                                min_order_param=0.0, max_order_param=None, prob_idxs=None, with_uncertainty=True,
                                uncertainty_types=('variation_ratio', 'predictive_entropy', 'mutual_information')):
    """ Aggregate structural transition data in order to plot it later.

    Starting from the results_file of the run_cnn_model function,
    aggregate the data by a given order parameter and the probabilities of
    each class.
    This is used to prepare the data for the structural transition plots,
    as shown in Fig. 4, Ziletti et al., Nature Communications 9, 2775 (2018).

    Parameters:

    filename: string,
        Full path to the results_file created by the run_cnn_model function.
        This is a csv file

    nb_samples: int
        Number of samples present in results_file for each order parameter step.

    nb_order_param_steps: int
        Number of order parameter steps. For example, if we are interpolating
        between structure_1 and structure_2 with 10 steps, nb_order_param_steps=10.

    max_order_param: float
        Maximum number that the order parameter will take in the dataset.
        This is used to create (together with nb_order_param_steps) to create
        the linear space which will be later used by the plotting function.

    prob_idxs: list of int
        List of integers which correspond to the classes for which the
        probabilities will be extracted from the results_file.
        prob_idxs=[0, 3] will extract only prob_predictions_0 and
        prob_predictions_3 from the results_file.

    Returns:

    panda dataframe
        A panda dataframe with the following columns:

        - a_to_b_index_ : value of the order parameter

        - 2i columns (where the i's are the elements of the list prob_idxs)
        as below:

            prob_predictions_i_mean : mean of the distribution of classification
            probability i for the given a_to_b_index_ value of the order parameter.

            prob_predictions_i_std : standard deviation of the distribution
            of classification probability i for the given a_to_b_index_
            value of the order parameter.

        - [optional]: columns containing uncertainty quantification

     .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    df = pd.read_csv(filename)

    # throw away first 'nb_rows_to_cut' rows because they come from descriptor_all_classes_8_samples.tar.gz
    # it is a workaround to have the neural network to predict even if not
    # all classes are present in the dataset
    df = df[nb_rows_to_cut:]

    # nb samples for each order parameter steps
    steps, step = np.linspace(min_order_param, max_order_param, nb_order_param_steps, retstep=True)
    a_to_b_index = np.repeat(steps, nb_samples)
    df['a_to_b_index'] = a_to_b_index

    prob_predictions = []
    prob_pred_agg = {}

    for prob_idx in prob_idxs:
        prob_prediction = 'prob_predictions_' + str(prob_idx)
        prob_predictions.append(prob_prediction)
        prob_pred_agg.update({prob_prediction: ['mean', 'std']})

    df_results_prob = df.groupby(['a_to_b_index'], as_index=False).agg(prob_pred_agg)

    # flatten hierarchical index
    # NB: you cannot just rename the columns
    # the values are ordered by increasing mean, so the column name --> value
    # will not be conserved
    df_results_prob.columns = ['_'.join(col).strip() for col in df_results_prob.columns.values]
    df_results_prob.reindex(columns=sorted(df_results_prob.columns))

    if with_uncertainty:
        uncertainty_preds = []
        uncertainty_pred_agg = {}

        for uncertainty_type in uncertainty_types:
            uncertainty_pred = 'uncertainty_' + str(uncertainty_type)
            uncertainty_preds.append(uncertainty_pred)
            uncertainty_pred_agg.update({uncertainty_pred: ['mean', 'std']})

        df_results_uncertainty = df.groupby(['a_to_b_index'], as_index=False).agg(uncertainty_pred_agg)
        df_results_uncertainty.columns = ['_'.join(col).strip() for col in df_results_uncertainty.columns.values]
        df_results_uncertainty.reindex(columns=sorted(df_results_uncertainty.columns))
        # df_results_uncertainty.drop('a_to_b_index_', axis=1, inplace=True)

    if with_uncertainty:
        # merge the probability prediction results with the uncertainty results
        df_results = pd.merge(df_results_prob, df_results_uncertainty, on='a_to_b_index_')
    else:
        df_results = df_results_prob

    return df_results


def make_crossover_plot(df_results, filename, filename_suffix, title, labels, nb_order_param_steps,
                        plot_type='probability', prob_idxs=None, uncertainty_type='mutual_information',
                        linewidth=1.0, markersize=1.0, max_nb_ticks=None, palette=None, show_plot=False,
                        style='publication', x_label="Order parameter"):
    """ Starting from an aggregated data panda dataframe, plot classification
    probability distributions as a function of an order parameter.

    This will produce a plot along the lines of Fig. 4, Ziletti et al.

    Parameters:

    df_results: panda dataframe,
        Panda dataframe returned by the `aggregate_struct_trans_data` function.

    filename: string
        Full path to the results_file created by the run_cnn_model function.
        This is a csv file. Only used to name the generated plot appriately.

    filename_suffix: string
        Suffix to be put for the plot filename. This suffix will determine
        the format of the output plot (e.g. '.png' or '.svg' will create
        a png or an svg file, respectively.)

    title: string
        Title of the plot

    plot_type: str (options: 'probability', 'uncertainty')
        Plot either probabilities of classification or uncertainty.

    uncertainty_type: str (options: 'mutual_information', 'predictive_entropy')
        Type of uncertainty estimation to be plotted. Used only if `plot_type`='uncertainty'.

    prob_idxs: list of int
        List of integers which correspond to the classes for which the
        probabilities will be extracted from the results_file.
        prob_idxs=[0, 3] will extract only prob_predictions_0 and
        prob_predictions_3 from the results_file.
        They should correspond (or be a subset) of the prob_idxs
        specified in aggregate_struct_trans_data.

    nb_order_param_steps: int
        Number of order parameter steps. For example, if we are interpolating
        between structure_1 and structure_2 with 10 steps, nb_order_param_steps=10.
        Must be the same as specified in aggregate_struct_trans_data.
        Different values might work, but could give rise to unexpected
        behaviour.

    show_plot: bool, optional, default: False
        If True, it opens the generated plot.

    style: string, optional, {'publication'}
        If style=='publication', load the default matplotlib style (white
        background).
        Otherwise, use the 'fivethirtyeight' matplotlib style (black background).
        plt.style.use('fivethirtyeight')

    x_label: string, optional, default: "Order parameter"
        Label for the x-axis (the order parameter axis)

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if style == 'publication':
        plt.style.use('default')
    else:
        plt.style.use('fivethirtyeight')

    # colors from https://matplotlib.org/examples/color/named_colors.html
    if palette is None:
        palette = ['yellow', 'red', 'blue', 'green', 'purple', 'orange', 'black']

    a_to_b_param = df_results.a_to_b_index_.values

    colors_plot = []
    labels_sel = []

    y_label_name_mean = []
    y_label_name_std = []

    if plot_type == 'probability':
        for prob_idx in prob_idxs:
            y_label_name_mean.append('prob_predictions_' + str(prob_idx) + '_mean')
            y_label_name_std.append('prob_predictions_' + str(prob_idx) + '_std')

            colors_plot.append(palette[prob_idx])
            labels_sel.append(labels[prob_idx])
    elif plot_type == 'uncertainty':
        y_label_name_mean.append('uncertainty_' + str(uncertainty_type) + '_mean')
        y_label_name_std.append('uncertainty_' + str(uncertainty_type) + '_std')

        colors_plot.append(palette[0])
        labels_sel.append(labels[0])
    else:
        raise Exception("Please specify a valid plot_type. Possible values are: 'probability', 'uncertainty'.")

    y_value_mean = []
    y_value_std = []

    if plot_type == 'probability':
        # a is 1st prob_idx, b is 2nd (order matters for the plot)
        for prob_idx in range(len(prob_idxs)):
            y_value_mean.append(df_results[y_label_name_mean[prob_idx]].values)
            y_value_std.append(df_results[y_label_name_std[prob_idx]].values)
    elif plot_type == 'uncertainty':
        y_value_mean.append(df_results[y_label_name_mean].values)
        y_value_std.append(df_results[y_label_name_std].values)
    else:
        pass

    # set max nb ticks
    if max_nb_ticks is not None:
        max_nb_ticks = min(max_nb_ticks, nb_order_param_steps)
    else:
        max_nb_ticks = nb_order_param_steps

    steps, step = np.linspace(np.amin(a_to_b_param), np.amax(a_to_b_param), max_nb_ticks, retstep=True)

    # the sigma/STD_SCALING upper and lower analytic population bounds
    std_scaling = 1.0
    lower_bound = []
    upper_bound = []

    if plot_type == 'probability':
        for prob_idx in range(len(prob_idxs)):
            lower_bound.append(y_value_mean[prob_idx] - y_value_std[prob_idx] / std_scaling)
            upper_bound.append(y_value_mean[prob_idx] + y_value_std[prob_idx] / std_scaling)
    elif plot_type == 'uncertainty':
        lower_bound.append(y_value_mean[0] - y_value_std[0] / std_scaling)
        upper_bound.append(y_value_mean[0] + y_value_std[0] / std_scaling)
    else:
        pass

    fig, ax = plt.subplots(1)

    plt.suptitle(title, fontname='Ubuntu', fontsize=15, fontstyle='italic', fontweight='bold')
    plt.tight_layout(pad=5.0, w_pad=2.0, h_pad=1.0)

    # restore defaults to 1.5.1 for reproducibility
    # https: // matplotlib.org / users / dflt_style_changes.html  # grid-lines
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    # ax.set_xlim([-np.amax(a_to_b_param) * 0.05 + np.amin(a_to_b_param), np.amax(a_to_b_param) * 1.05])
    ax.set_xlim([np.amin(a_to_b_param), np.amax(a_to_b_param)])

    if plot_type == 'probability':
        ax.set_ylim([-0.1, 1.1])
    start, end = ax.get_xlim()

    ax.xaxis.set_ticks(steps)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(2)
        # specify integer or one of preset strings, e.g.
        # tick.label.set_fontsize('x-small')
        tick.label.set_rotation('vertical')

    if plot_type == 'probability':
        for prob_idx in range(len(prob_idxs)):
            ax.plot(a_to_b_param, y_value_mean[prob_idx], marker='o', linestyle='-', color=colors_plot[prob_idx],
                    label=labels_sel[prob_idx], linewidth=linewidth, markeredgecolor=colors_plot[prob_idx],
                    markersize=markersize)
            ax.fill_between(a_to_b_param, lower_bound[prob_idx], upper_bound[prob_idx], facecolor=colors_plot[prob_idx],
                            alpha=0.2, edgecolor=colors_plot[prob_idx], linewidth=0.0)
    elif plot_type == 'uncertainty':
        ax.plot(a_to_b_param, y_value_mean[0], marker='o', linestyle='-', color=colors_plot[0],
                label=labels_sel[0], linewidth=linewidth, markeredgecolor=colors_plot[0],
                markersize=markersize)
        ax.fill_between(a_to_b_param, np.array(lower_bound).reshape(-1), np.array(upper_bound).reshape(-1)
                        , facecolor=colors_plot[0],
                        alpha=0.2, edgecolor=colors_plot[0], linewidth=0.0)
    else:
        pass

    ax.set_xlabel(x_label, fontsize=15)

    if plot_type == 'probability':
        ax.set_ylabel("Classification probability", fontsize=15)
    elif plot_type == 'uncertainty':
        if uncertainty_type == 'mutual_information':
            ax.set_ylabel("Mutual information", fontsize=15)
        elif uncertainty_type == 'predictive_entropy':
            ax.set_ylabel("Predictive entropy", fontsize=15)
        else:
            ax.set_ylabel("Label", fontsize=15)

    ax.tick_params(labelsize=15)

    legend = ax.legend(loc='center left', fontsize=10, bbox_to_anchor=(0.1, 0.5), borderaxespad=1.0, frameon=1)

    if style == 'publication':
        for text in legend.get_texts():
            plt.setp(text, color=(0 / 255, 0 / 255, 0 / 255))
    else:
        for text in legend.get_texts():
            plt.setp(text, color=(224 / 255, 224 / 255, 224 / 255))

        ax.set_axis_bgcolor((224 / 255, 224 / 255, 224 / 255))

        frame = legend.get_frame()
        frame.set_facecolor((32 / 255, 32 / 255, 32 / 255))
        frame.set_edgecolor((32 / 255, 32 / 255, 32 / 255))

    if filename_suffix == ".png":
        plt.savefig(filename.rsplit('.', 1)[0] + '_' + plot_type + filename_suffix, format="png")
    elif filename_suffix == ".svg":
        plt.savefig(filename.rsplit('.', 1)[0] + '_' + plot_type + filename_suffix, format="svg")
    else:
        raise Exception("Filename suffix {0} is not a valid file format.".format(filename_suffix))

    if show_plot:
        plt.show()


def show_images(images, filename_png, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Taken from https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window

    Parameters:

    images: list of np.arrays
        Images to be plotted. It must be compatible with plt.imshow.

    cols: int,  optional, (default = 1)
        Number of columns in figure (number of rows is
        set to np.ceil(n_images/float(cols))).

    titles: list of strings
        List of titles corresponding to each image.

    """
    plt.clf()
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image, interpolation='spline16', cmap='viridis', vmin=np.amin(images), vmax=np.amax(images))
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.savefig(filename_png, dpi=10, format='png')


def make_multiple_image_plot(data, title="Figure 1", cmap=cm.hot, n_rows=None, n_cols=None, vmin=None,
                             vmax=None, filename=None, save=False):
    fig = plt.figure()
    plt.suptitle(title, fontname='Ubuntu', fontsize=15, fontstyle='italic', fontweight='bold')

    plt.style.use('fivethirtyeight')

    margin = 0.08
    w = (1.0 - margin * 2) / n_cols
    h = (1.0 - margin * 2) / n_rows

    nb_channels = data.shape[1]

    cmaps = []
    if nb_channels == 3:
        # define colormaps
        # from 0 to full red, green and blue
        colors_for_maps = ["red", "green", "blue"]
        for color_for_maps in colors_for_maps:
            cmaps.append(rgb_colormaps(color_for_maps))
    elif nb_channels == 1:
        cmaps.append(cmap)
    else:
        raise Exception("Unexpected number of color channels: {}".format(nb_channels))

    filenames_ch = []
    for idx_ch in range(nb_channels):
        images = []
        idx_filter = 0
        for i in range(n_cols):
            for j in range(n_rows):
                if idx_filter < data.shape[0]:
                    # https://python4astronomers.github.io/plotting/advanced.html
                    # bottom first
                    # pos = [margin + i*1.0*w, margin + j*1.0*h, w, h]
                    # top first
                    pos = [margin + i * 1.0 * w, (1.0 - j * 1.0 * h - h - margin), w, h]
                    a = fig.add_axes(pos)

                    data_filter = data[idx_filter, idx_ch, :, :]
                    dd = ravel(data_filter)
                    # Manually find the min and max of all colors for
                    # use in setting the color scale.
                    vmin = min(vmin, amin(dd))
                    # make sure vmin is positive or zero
                    vmin = max(0.0, vmin)

                    # stretches the images to the desired width
                    images.append(a.imshow(data_filter, cmap=cmaps[idx_ch], vmin=vmin, vmax=vmax))

                    # do not show axis
                    plt.axis('off')

                    idx_filter += 1

        # split filename to remove path from extension
        filename_no_ext, file_extension = os.path.splitext(filename)
        filename_ch = filename_no_ext + "_ch" + str(idx_ch) + file_extension
        filenames_ch.append(filename_ch)

        if save:
            logger.info("Saving multiple image plot to file.")
            logger.debug("Filename: {0}".format(filename))
            plt.savefig(filename_ch, dpi=600, format="png")

    plt.clf()

    return filenames_ch


def rgb_colormaps(color):
    """Obtain colormaps for RGB.

    For a general overview: https://matplotlib.org/examples/pylab_examples/custom_cmap.html"""

    if color == "red":
        cdict = {'red': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),

                 'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    elif color == "green":
        cdict = {'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),

                 'green': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),

                 'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    elif color == "blue":
        cdict = {'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),

                 'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}
    else:
        raise ValueError("Wrong color specified. Allowed colors are 'red', 'green', 'blue'.")

    cmap = LinearSegmentedColormap('BlueRed2', cdict)

    return cmap


def plot_confusion_matrix(conf_matrix, classes, conf_matrix_file, normalize=False, title='Confusion matrix',
                          title_true_label='True label', title_pred_label='Predicted label', cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        logger.debug("Normalized confusion matrix")
    else:
        logger.debug('Confusion matrix, without normalization')

    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    # add this otherwise the x-axis gets cut
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.ylabel(title_true_label)
    plt.xlabel(title_pred_label)
    # plt.show()
    plt.savefig(conf_matrix_file, dpi=100, format="png")
    plt.clf()
