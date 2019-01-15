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

import json
import datetime
import logging
import hashlib
import os
import numpy as np
import numbers
import pandas as pd
from jinja2 import Template
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, Rect
from six.moves import zip
from sklearn import preprocessing
from bokeh.models import Circle
from ai4materials.utils.utils_crystals import format_e
from ai4materials.utils.utils_crystals import create_supercell
from scipy.spatial import ConvexHull
from ai4materials.utils.utils_config import copy_directory

logger = logging.getLogger('ai4materials')


class Viewer(object):
    """Interactively visualize - possibly large - materials science datasets."""

    def __init__(self, title=None, name=None, width=600, height=700, configs=None):

        if name is None:
            now = datetime.datetime.now()
            name = hashlib.sha224(str(now).encode('utf-8')).hexdigest()[:16]

        self.title = title
        self.width = width
        self.height = height
        self.name = name
        self.configs = configs

        palette_classification = ['#000000', '#0072B2', '#009E73', '#E69F00', '#CC79A7', '#f44336', '#e91e63',
                                  '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688',
                                  '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722',
                                  '#795548', '#9e9e9e', '#607d8b', '#b71c1c', '#880e4f', '#4a148c', '#311b92',
                                  '#1a237e', '#0d47a1', '#01579b', '#006064', '#004d40', '#1b5e20', '#33691e',
                                  '#827717', '#f57f17', '#ff6f00', '#e65100', '#bf360c', '#3e2723', '#212121',
                                  '#263238']

        # color-blind safe (http://mkweb.bcgsc.ca/colorblind/)
        palette_regression = ['#000000', '#0072B2', '#009E73', '#E69F00', '#CC79A7']
        self.palette_classification = palette_classification
        self.palette_regression = palette_regression

        # specifications for interactive plot
        title_text = dict(font="Trebuchet MS", font_size='30pt', color='#20335d', font_style='bold', baseline='bottom')
        xaxis = dict(axis_label_text_font_size="14pt")
        yaxis = dict(axis_label_text_font_size="14pt")
        self.title_text = title_text
        self.xaxis = xaxis
        self.yaxis = yaxis

        # specifications for small plot (the map plot on the right hand side)
        title_text_map = dict(font="Trebuchet MS", font_size='18pt', color='#20335d', font_style='bold',
                              baseline='bottom')
        self.title_text_map = title_text_map

    def plot_with_structures(self, ase_atoms_list=None, x=None, y=None, outlog_file=None, is_classification=True,
                             target=None, target_pred=None, target_unit=None, target_name=None,
                             create_replicas_by='user-defined', target_replicas=(3, 3, 3), descriptor=None,
                             legend_title=None, x_axis_label=None, y_axis_label=None, plot_title='Interactive plot',
                             point_size=None, map_point_size=None, tmp_folder=None, show_convex_hull=False):
        """ Plot the NOMAD Viewer given list of atomic structures.

        Parameters:

        ase_atoms_list: list of ``ASE atoms`` objects
            List of structures.

        colors: string, list
            Color palette used for the points in the plot.
            For regression: hardcoded to 5 values. Each color represents a quintile.
            We use percentile to have a scale which is robust to outliers.
            For classification: hardcoded to a maximum of 37 values. Each color is associated to a unique class value.

        x: list of floats
            First coordinates (x) of the interactive plot in the Viewer.

        y: list of floats
            Second coordinates (x) of the interactive plot in the Viewer.

        target: list of floats, optional
            Scalar quantity to be predicted. It is used to define the colors of the points in the plot.

        target_pred: list of floats, optional
            Scalar quantity to be predicted by the underlying model.
            It is compared with ``target`` is the case of regression to show the error made by the model on
            each sample. It is not used to color the points; ``target`` is used for the colors of the points
            in the plot.

        target_class_names: list of floats or strings, optional
            List of classes for the case of classification.

        x_axis_label: string, optional
            Label of the x-axis.

        y_axis_label: string, optional
            Label of the y-axis.

        point_size: float, optional (default=12)
            Size of the points (in pt) in the Viewer interactive plot.

        map_point_size: float, optional (default=``point_size``*0.75)
            Size of the points (in pt) in the map plot. The map plot is the small plot on the right hand side of the
            Viewer page

        tmp_folder: string, optional (default=``configs['io']['tmp_folder']``)
            Path to the temporary folder where the images, the input files,
            the descriptor files, and the similarity matrix are written.

        show_convex_hull: bool, default False
            calculate and plot convex hull for each color partition (class in case of classification)

        Returns:

        file_html_link: string
            html string that in the Beaker notebook generates the html link to the
            viewer (name.html). For example,
            <a target=_blank href='/path/to/file/viewer.html'>Click here to open the Viewer</a>

        file_html_name: string
            Absolute path to where the NOMAD Viewer (html page) is generated.

        """

        if target is None:
            raise ValueError("Please pass a target vector. "
                             "This is an array of scalar values that it is used to color the plot. ")

        if target_unit is not None:
            target_unit_legend = target_unit
        else:
            target_unit = 'arb. units'
            target_unit_legend = ''

        if target_name is None:
            target_name = 'Target'

        if legend_title is None:
            legend_title = ''

        if point_size is None:
            point_size = 12

        if map_point_size is None:
            map_point_size = point_size * 0.75

        if x_axis_label is not None:
            if y_axis_label is not None:
                show_axis = True
        else:
            show_axis = False

        if tmp_folder is None:
            tmp_folder = self.configs['io']['tmp_folder']

        # read output log and substitute html end of line characters
        if outlog_file is not None:
            try:
                outf = open(outlog_file, 'r')
                outf_string = str(outf.read()).replace("\n", "<br>")
            except IOError as err:
                logger.info("Output file not found. {}".format(err))
                outf_string = ''
        else:
            outf_string = ''

        # make supercell if structure is periodic
        ase_atoms_list = [
            create_supercell(ase_atoms, create_replicas_by=create_replicas_by, target_replicas=target_replicas) if np.all(ase_atoms.pbc) else ase_atoms for
            ase_atoms in ase_atoms_list]

        # copy jsmol folder and create thumbnails
        copy_jsmol(configs=self.configs, tmp_folder=tmp_folder)
        ase_atoms_list = write_thumbnail_files(ase_atoms_list, configs=self.configs)
        ase_atoms_list = write_geo_files(ase_atoms_list, configs=self.configs)

        names = [atoms.info['label'] for atoms in ase_atoms_list]
        chemical_formulas = [atoms.get_chemical_formula(mode='hill') for atoms in ase_atoms_list]
        geo_files = [atoms.info['geometry_filename'] for atoms in ase_atoms_list]
        png_files = [atoms.info['png_filename'] for atoms in ase_atoms_list]

        # HARD-CODED for binary dataset
        # chemical_formulas = rename_material(chemical_formulas)

        # define a default jsmol window
        which_jsmol = np.zeros(len(x))

        # quantile-based discretization function
        # discretize variable into equal-sized buckets to have a color scale which is robust to outliers
        target = np.asarray(target)
        colors, classes, bins, n_quantiles = self._get_colors(is_classification=is_classification, target=target)

        # put the necessary info in a ColumnDataSource to use in the plot
        # ColumnDataSource is used by Bokeh (the library for the interactive plot)

        # truncation to a given number of significative digits to have a nicer plot
        target_hover = [format_e(item) if isinstance(item, numbers.Number) else target_pred for item in target]

        data_main_plot = dict(x=x, y=y, jmol_file=[], name=names, chemical_formula=chemical_formulas,
                              target=target_hover, colors=colors, whichJSmol=which_jsmol, geo_file=geo_files,
                              imgs=png_files)

        # add variables if they are not None
        if target_pred is not None:
            target_pred_hover = [format_e(item) for item in target_pred if isinstance(item, numbers.Number)]
            data_main_plot['target_pred'] = target_pred_hover

        if target_pred is not None and target is not None:
            if not is_classification:
                abs_error = [abs(target_i - target_pred_i) for target_i, target_pred_i in zip(target, target_pred)]
                abs_error_hover = [format_e(item) for item in abs_error]
                data_main_plot['abs_error'] = abs_error_hover

        source = ColumnDataSource(data=data_main_plot)

        # ColumnDataSource to use in the 'Map' plot
        # NOTE: we do not use the same because otherwise the glyph selection properties are passed automatically
        # and we loose the colors in the Map plot when a point is selected
        x_zoom = np.zeros(len(x))
        y_zoom = np.zeros(len(y))
        width_zoom = np.zeros(len(x))
        height_zoom = np.zeros(len(x))

        source_map = ColumnDataSource(
            data=dict(x=x, y=y, colors=colors, x_zoom=x_zoom, y_zoom=y_zoom, width_zoom=width_zoom,
                      height_zoom=height_zoom))

        # different Hover tool different according to the task performed
        if target_pred is not None:
            tooltips = load_templates('tooltip_pred_target').format(target_name, target_unit)
        else:
            tooltips = load_templates('tooltip_unsupervised').format(target_name, target_unit)

        hover = HoverTool(tooltips=tooltips)

        tools_main_plot = hover, "wheel_zoom,box_zoom,pan,reset,tap,previewsave,resize"
        p1 = self._make_main_plot(tools=tools_main_plot, plot_title=plot_title, colors=colors,
                                 show_axis=show_axis,
                                 source=source, point_size=point_size, x_axis_label=x_axis_label,
                                 y_axis_label=y_axis_label)

        tools_small_plot = "pan,box_zoom,wheel_zoom,resize,reset"
        p2 = self._make_small_plot(tools=tools_small_plot, point_size=map_point_size, colors=colors,
                                  source_map=source_map, source=source)

        if show_convex_hull:
            p1, p2 = plot_convex_hull(colors=colors, x=x, y=y, p1=p1, p2=p2)

        # JS code to be used in the callback to load the corresponding structure in JSmol
        # when user clicks on a point of the main plot
        js_load_jmol_1 = load_js_scripts('js_load_jmol_1')

        if target_pred is not None:
            js_load_jmol_2 = load_js_scripts('js_load_jmol_2_pred')
        else:
            js_load_jmol_2 = load_js_scripts('js_load_jmol_2_no_pred')

        js_load_jmol_3 = load_js_scripts('js_load_jmol_3')

        if target_pred is not None:
            js_load_jmol_4 = load_js_scripts('js_load_jmol_4_pred')
        else:
            js_load_jmol_4 = load_js_scripts('js_load_jmol_4_no_pred')

        js_load_jmol_5 = load_js_scripts('js_load_jmol_5')

        js_load_jmol = js_load_jmol_1 + js_load_jmol_2 + js_load_jmol_3 + js_load_jmol_4 + js_load_jmol_5

        # returns the TapTool objects of p1 (main plot)
        taptool = p1.select(type=TapTool)

        # load the corresponding crystal structure when a point on the main plot is clicked
        # load in either 1st or 2nd JSmol applet
        taptool.callback = CustomJS(args=dict(source=source), code=js_load_jmol)

        # plots can be a single Bokeh model, a list/tuple, or even a dictionary
        plots = {'main_plot': p1, 'Map': p2}

        script, div = components(plots)

        # template for the HTML page to be generated
        html_viewer_head = load_templates('html_page_head_1').format(
            str(self.configs['html']['css_file_viewer'])) + load_templates('html_page_head_2')

        if target_pred is not None:
            write_summary_function = load_js_scripts('write_summary_function_pred')
        else:
            write_summary_function = load_js_scripts('write_summary_function_no_pred')

        html_template_viewer_2 = load_templates('html_template_viewer_2')

        legend = _make_legend(legend_title=legend_title, is_classification=is_classification,
                              target_class_names=classes, n_quantiles=n_quantiles, bins=bins,
                              target_unit_legend=target_unit_legend)

        html_template_viewer_2_1 = load_templates('html_viewer_instructions_with_jsmol')

        if target_pred is not None:
            html_template_viewer_3 = load_templates('html_recap_table_header_pred_target').format(target_name,
                                                                                                  target_unit)
        else:
            html_template_viewer_3 = load_templates('html_recap_table_header_no_pred_target').format(target_name,
                                                                                                     target_unit)

        html_template_viewer_4 = load_templates('html_recap_table_clear_selection').format(outf_string)

        template = Template(
            html_viewer_head + write_summary_function + html_template_viewer_2 + legend + html_template_viewer_2_1 +
            html_template_viewer_3 + html_template_viewer_4)

        # javascript script to be included in the HTML page to load JSmol
        js_jsmol = load_js_scripts('load_jsmol_applet')

        # output static HTML file
        # with Beaker only certain files are accessible by the browsers
        # in particular, only webpage in "/home/beaker/.beaker/v1/web/" and subfolders can be accessed
        if self.configs['runtime']['isBeaker']:
            # if Beaker is used
            file_html_name = '/home/beaker/.beaker/v1/web/tmp/' + self.name + '.html'
            file_html_link = "<a target=_blank href='/user/tmp/" + self.name + ".html'> Click here to open the Viewer </a>"

            logger.info("Click on the button 'View interactive 2D scatter plot' to see the plot.")
        else:
            file_html_name = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, '{}.html'.format(self.name))))
            file_html_link = None

        # build the page HTML
        html = template.render(js_resources=js_jsmol, script=script, div=div)
        with open(file_html_name, 'w') as f:
            f.write(html)
            f.flush()
            f.close()

        logging.info("NOMAD Viewer saved at: {}".format(file_html_name))

        return file_html_link, file_html_name

    def plot(self, x=None, y=None, outlog_file=None, is_classification=True, target=None, target_pred=None,
             target_unit=None, target_name=None, png_files=None,
             descriptor=None, legend_title=None, x_axis_label=None, y_axis_label=None,
             plot_title='Interactive plot', point_size=None, map_point_size=None, tmp_folder=None,
             show_convex_hull=None, df_tooltip=None):
        """ Plot the Viewer given list of data points.

        Parameters:

        colors: string, list
            Color palette used for the points in the plot.
            For regression: hardcoded to 5 values. Each color represents a quintile.
            We use percentile to have a scale which is robust to outliers.
            For classification: hardcoded to a maximum of 37 values. Each color is associated to a unique class value.

        x: list of floats
            First coordinates (x) of the interactive plot in the Viewer.

        y: list of floats
            Second coordinates (x) of the interactive plot in the Viewer.

        target: list of floats, optional
            Scalar quantity to be predicted. It is used to define the colors of the points in the plot.

        target_pred: list of floats, optional
            Scalar quantity to be predicted by the underlying model.
            It is compared with ``target`` is the case of regression to show the error made by the model on
            each sample. It is not used to color the points; ``target`` is used for the colors of the points
            in the plot.

        target_class_names: list of floats or strings, optional
            List of classes for the case of classification.

        x_axis_label: string, optional
            Label of the x-axis.

        y_axis_label: string, optional
            Label of the y-axis.

        point_size: float, optional (default=12)
            Size of the points (in pt) in the Viewer interactive plot.

        map_point_size: float, optional (default=``point_size``*0.75)
            Size of the points (in pt) in the map plot. The map plot is the small plot on the right hand side of the
            Viewer page

        tmp_folder: string, optional (default=``configs['io']['tmp_folder']``)
            Path to the temporary folder where the images, the input files,
            the descriptor files, and the similarity matrix are written.

        show_convex_hull: bool, default False
            calculate and plot convex hull for each color partition (class in case of classification)

        Returns:

        file_html_link: string
            html string that in the Beaker notebook generates the html link to the
            viewer (name.html). For example,
            <a target=_blank href='/path/to/file/viewer.html'>Click here to open the Viewer</a>

        file_html_name: string
            Absolute path to where the NOMAD Viewer (html page) is generated.

        """

        if target is None:
            raise ValueError("Please pass a target vector. "
                             "This is an array of scalar values that it is used to color the plot. ")

        if target_unit is not None:
            target_unit_legend = target_unit
        else:
            target_unit = 'arb. units'
            target_unit_legend = ''

        if target_name is None:
            target_name = 'Target'

        if legend_title is None:
            legend_title = ''

        if point_size is None:
            point_size = 12

        if map_point_size is None:
            map_point_size = point_size * 0.75

        show_axis = False
        if x_axis_label is not None:
            if y_axis_label is not None:
                show_axis = True

        if tmp_folder is None:
            tmp_folder = self.configs['io']['tmp_folder']

        # read output log and substitute html end of line characters
        if outlog_file is not None:
            outf = open(outlog_file, 'r')
            outf_string = str(outf.read()).replace("\n", "<br>")
        else:
            outf_string = ''

        # quantile-based discretization function
        # discretize variable into equal-sized buckets to have a color scale which is robust to outliers
        target = np.asarray(target)
        df_target = pd.DataFrame(target, columns=['target'])

        colors, classes, bins, n_quantiles = self._get_colors(is_classification=is_classification, target=target)

        # put the necessary info in a ColumnDataSource to use in the plot
        # ColumnDataSource is used by Bokeh (the library for the interactive plot)

        # truncation to a given number of significative digits to have a nicer plot
        target_hover = [format_e(item) if isinstance(item, numbers.Number) else target_pred for item in target]

        data_main_plot = dict(x=x, y=y, target=target_hover, colors=colors)

        cols_to_show_tooltip = []

        # add variables if they are not None
        if target_pred is not None:
            target_pred_hover = [format_e(item) for item in target_pred if isinstance(item, numbers.Number)]
            data_main_plot['target_pred'] = target_pred_hover
            cols_to_show_tooltip.append(target_pred)

        if target_pred is not None and target is not None:
            if not is_classification:
                abs_error = [abs(target_i - target_pred_i) for target_i, target_pred_i in zip(target, target_pred)]
                abs_error_hover = [format_e(item) for item in abs_error]
                data_main_plot['abs_error'] = abs_error_hover
                cols_to_show_tooltip.append(abs_error_hover)

        if png_files is not None:
            data_main_plot['imgs'] = png_files

        # add data from dataframe
        if df_tooltip is not None:
            for col in list(df_tooltip.columns.values):
                if ' ' in col:
                    logging.warning("Spaces in features for Viewer tooltip are not allowed")
                    logging.warning("Replacing ' ' with '_' in feature: {}".format(col))
                    col_tooltip = col.replace(' ', '_')
                else:
                    col_tooltip = col

                data_main_plot[col_tooltip] = df_tooltip[col]
                cols_to_show_tooltip.append(col_tooltip)

        source = ColumnDataSource(data=data_main_plot)

        # different Hover tool different according to the task performed
        if 'imgs' in data_main_plot.keys():
            if target_pred is not None:
                tooltips = load_templates('tooltip_pred_target').format(target_name, target_unit)
            else:
                tooltips = load_templates('tooltip_unsupervised').format(target_name, target_unit)
        else:
            tooltips = [tuple([str(col), '@' + str(col)]) for col in cols_to_show_tooltip]

        hover = HoverTool(tooltips=tooltips)

        # ColumnDataSource to use in the 'Map' plot
        # NOTE: we do not use the same because otherwise the glyph selection properties are passed automatically
        # and we loose the colors in the Map plot when a point is selected
        # initialize the zoom window to zero
        x_zoom = np.zeros(len(x))
        y_zoom = np.zeros(len(y))
        width_zoom = np.zeros(len(x))
        height_zoom = np.zeros(len(x))

        source_map = ColumnDataSource(
            data=dict(x=x, y=y, colors=colors, x_zoom=x_zoom, y_zoom=y_zoom, width_zoom=width_zoom,
                      height_zoom=height_zoom))

        tools_main_plot = hover, "wheel_zoom,box_zoom,pan,reset,tap,previewsave,resize"
        p1 = self._make_main_plot(tools=tools_main_plot, plot_title=plot_title, colors=colors, show_axis=show_axis,
                                 source=source, point_size=point_size, x_axis_label=x_axis_label,
                                 y_axis_label=y_axis_label)

        tools_small_plot = "pan,box_zoom,wheel_zoom,resize,reset"
        p2 = self._make_small_plot(tools=tools_small_plot, point_size=map_point_size, colors=colors,
                                  source_map=source_map, source=source)

        if show_convex_hull:
            p1, p2 = plot_convex_hull(colors=colors, x=x, y=y, p1=p1, p2=p2)

        # plots can be a single Bokeh model, a list/tuple, or even a dictionary
        plots = {'main_plot': p1, 'Map': p2}
        script, div = components(plots)

        # template for the HTML page to be generated
        html_viewer_head = load_templates('html_page_head_1').format(
            str(self.configs['html']['css_file_viewer'])) + load_templates('html_page_head_2')

        if target_pred is not None:
            write_summary_function = load_js_scripts('write_summary_function_pred')
        else:
            write_summary_function = load_js_scripts('write_summary_function_no_pred')

        html_template_viewer_2 = load_templates('html_template_viewer_2')

        legend = _make_legend(legend_title=legend_title, is_classification=is_classification,
                              target_class_names=classes, n_quantiles=n_quantiles, bins=bins,
                              target_unit_legend=target_unit_legend)

        html_template_viewer_2_1 = load_templates('html_viewer_instructions')

        html_template_viewer_3 = load_templates('show_outfile').format(outf_string)

        template = Template(
            html_viewer_head + write_summary_function + html_template_viewer_2 + legend + html_template_viewer_2_1
            + html_template_viewer_3)

        # output static HTML file
        # with Beaker only certain files are accessible by the browsers
        # in particular, only webpage in "/home/beaker/.beaker/v1/web/" and subfolders can be accessed
        if self.configs['runtime']['isBeaker']:
            # if Beaker is used
            file_html_name = '/home/beaker/.beaker/v1/web/tmp/' + self.name + '.html'
            file_html_link = "<a target=_blank href='/user/tmp/" + self.name + ".html'> Click here to open the Viewer </a>"

            logger.info("Click on the button 'View interactive 2D scatter plot' to see the plot.")
        else:
            file_html_name = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, '{}.html'.format(self.name))))
            file_html_link = None

        # build the page HTML
        html = template.render(script=script, div=div)
        with open(file_html_name, 'w') as f:
            f.write(html)
            f.flush()
            f.close()

        logging.info("NOMAD Viewer saved at: {}".format(file_html_name))

        return file_html_link, file_html_name

    def _get_colors(self, is_classification, target):

        classes = None
        bins = None
        n_quantiles = None

        # get number of unique target values
        if is_classification:

            le = preprocessing.LabelEncoder()
            le.fit(target)
            classes = (list(le.classes_))
            target_encoded = le.transform(target)

            n_classes = len(list(set(target)))
            colors = [self.palette_classification[item] for item in target_encoded]

            if n_classes > len(self.palette_classification):
                raise ValueError("You have more classes than available colors. \n"
                                 "Available colors: {}. Classes in the dataset: {}".format(
                                     len(self.palette_classification), n_classes))

        else:
            df_target = pd.DataFrame(target, columns=['target'])

            # try to divide in 5 quantiles, if it does not work divide in less
            for i in range(5, 0, -1):
                try:
                    target_bin = (pd.qcut(df_target['target'], i, labels=False)).values
                    logger.info('The color in the plot is given by the target value.')
                    bins = list((pd.qcut(df_target['target'], i, labels=False, retbins=True))[1])
                    #                    bins = np.around(bins, decimals=3)
                    bins = [format_e(item) for item in bins]
                    colors = [self.palette_regression[idx] for idx in target_bin]
                    n_quantiles = i
                    break
                except BaseException:
                    pass

        return colors, classes, bins, n_quantiles

    def _make_main_plot(self, tools, plot_title, colors, show_axis, source, point_size, x_axis_label=None,
                       y_axis_label=None):
        # Create a set of tools to use in the Bokeh plot

        # create main plot
        p1 = figure(title=plot_title, plot_width=600, plot_height=600, tools=tools, background_fill_color='#f2f2f2',
                    outline_line_width=0.01, toolbar_location="left")

        p1.title_text_font = self.title_text['font']
        p1.title_text_font_size = self.title_text['font_size']
        p1.title_text_color = self.title_text['color']
        p1.title_text_font_style = self.title_text['font_style']
        p1.title_text_baseline = self.title_text['baseline']

        if not show_axis:
            p1.axis.visible = None
            p1.xgrid.grid_line_color = None
            p1.ygrid.grid_line_color = None
        else:
            p1.axis.visible = True
            p1.xaxis.axis_label = x_axis_label
            p1.yaxis.axis_label = y_axis_label
            p1.xaxis.axis_label_text_font_size = self.xaxis['axis_label_text_font_size']
            p1.yaxis.axis_label_text_font_size = self.yaxis['axis_label_text_font_size']

        # JS code to reset the plot area according to the selection of the user
        p1.x_range.callback = CustomJS(args=dict(source=source),
                                       code=load_js_scripts('js_zoom') % ('x_zoom', 'width_zoom'))
        p1.y_range.callback = CustomJS(args=dict(source=source),
                                       code=load_js_scripts('js_zoom') % ('y_zoom', 'height_zoom'))

        # define the renderer and actually plot the point in figure p1 (main plot)
        r1 = p1.circle('x', 'y', size=point_size, fill_color=colors, fill_alpha=1.0, source=source, line_color=None,
                       nonselection_fill_alpha=0.1, nonselection_fill_color="blue", nonselection_line_color=None,
                       nonselection_line_alpha=0.0)

        return p1

    def _make_small_plot(self, tools, point_size, colors, source_map, source):
        # create small figure with the Map of the main plot
        p2 = figure(title='Map', plot_width=350, plot_height=300, tools=tools, background_fill_color="#ffffff",
                    outline_line_width=0.01, toolbar_location="right")

        p2.title_text_font = self.title_text_map['font']
        p2.title_text_font_size = self.title_text_map['font_size']
        p2.title_text_color = self.title_text_map['color']
        p2.title_text_font_style = self.title_text_map['font_style']
        p2.title_text_baseline = self.title_text_map['baseline']

        p2.axis.visible = None
        p2.xgrid.grid_line_color = None
        p2.ygrid.grid_line_color = None

        # define the renderer and actually plot the point in figure p2 (Map plot)
        r2 = p2.circle('x', 'y', size=point_size, fill_color=colors, fill_alpha=1.0, source=source_map, line_color=None)

        # r2.selection_glyph = Circle(fill_color='blue', line_color=None)
        r2.nonselection_glyph = Circle(fill_color='blue', fill_alpha=1.0, line_color=None)
        rect = Rect(x='x_zoom', y='y_zoom', width='width_zoom', height='height_zoom', fill_alpha=0.6, line_color=None,
                    fill_color='blue')
        # pass source (not source_map) otherwise the Box will not be shown on the Map plot
        p2.add_glyph(source, rect)

        return p2


def plot_convex_hull(colors, x, y, p1, p2):
    for color_of_class in list(set(colors)):
        matching_indices = [i for i, cc in enumerate(colors) if cc == color_of_class]
        x_y_array = np.array([(x[i], y[i]) for i in matching_indices])
        len_match_ind = len(matching_indices)
        if len_match_ind > 1:
            if len_match_ind > 2:
                hull = ConvexHull(x_y_array)
                x_for_hull, y_for_hull = x_y_array[hull.vertices].transpose()
            else:
                x_for_hull, y_for_hull = x_y_array.transpose()
            p1.patch(x_for_hull, y_for_hull, color=color_of_class, alpha=0.5)
            p2.patch(x_for_hull, y_for_hull, color=color_of_class, alpha=0.5)

    return p1, p2


def write_geo_files(ase_atoms_list, configs=None, dest_folder=None, format_geo='aims', filename_suffix='_aims.in'):
    """From a list of ASE atoms object, write a geometry file for each structure.

    ..todo:: add tests to check if it works with configs and dest folder as expected

    """

    if configs is None and dest_folder is None:
        raise Exception("Please specify either a config or a destination folder.")

    if configs is not None:
        dest_folder = configs['io']['tmp_folder']

    logger.info("Generating geometry files...")
    for atoms in ase_atoms_list:
        geo_filename = '{0}_op_0{1}'.format(atoms.info['label'], filename_suffix)
        geo_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, geo_filename)))
        # atoms.wrap()
        atoms.write(geo_filepath, format=format_geo)

        # for some reason Beaker needs a special path (not the actual path where the images are)
        # to visualize the image correctly
        if configs is not None:
            if configs['runtime']['isBeaker']:
                geo_filepath = os.path.abspath(os.path.normpath(os.path.join('/user/tmp/', geo_filename)))
            else:
                geo_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, geo_filename)))
        else:
            geo_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, geo_filename)))

        atoms.info['geometry_filename'] = geo_filepath

    logger.info("Done.")

    return ase_atoms_list


def write_thumbnail_files(ase_atoms_list, configs=None, dest_folder=None, filename_suffix='.png', rotation=True):
    """From a list of ASE atoms object, write a thumbnail based on the geometry to file for each structure.

    ..todo:: add tests to check if it works with configs and dest folder as expected

    """

    if configs is None and dest_folder is None:
        raise Exception("Please specify either a config or a destination folder.")

    if configs is not None:
        dest_folder = configs['io']['tmp_folder']

    logger.info("Generating thumbnail files...")
    for atoms in ase_atoms_list:
        png_filename = '{0}_op_0_geo_thumbnail{1}'.format(atoms.info['label'], filename_suffix)
        png_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, png_filename)))

        if rotation:
            rot = '10z,-80x'
        else:
            rot = '0x, 0y, 0z'

        kwargs = {'rotation': rot, 'radii': .50,  # float, or a list with one float per atom
                  'colors': None,  # List: one (r, g, b) tuple per atom
                  'show_unit_cell': 0,  # 0, 1, or 2 to not show, show, and show all of cell
                  'scale': 100, }

        atoms.write(png_filepath, format='png', **kwargs)

        # for some reason Beaker needs a special path (not the actual path where the images are)
        # to visualize the image correctly
        if configs is not None:
            if configs['runtime']['isBeaker']:
                png_filepath = os.path.abspath(os.path.normpath(os.path.join('/user/tmp/', png_filename)))
            else:
                png_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, png_filename)))
        else:
            png_filepath = os.path.abspath(os.path.normpath(os.path.join(dest_folder, png_filename)))

        atoms.info['png_filename'] = png_filepath

    logger.info("Done.")

    return ase_atoms_list


def read_control_file(control_file):
    """Check if there is a control file in order to read in info to be used for the Viewer"""

    x_axis_label = None
    y_axis_label = None

    try:
        with open(control_file) as data_file:
            data = json.load(data_file)

            for c in data['model_info']:
                x_axis_label = c["x_axis_label"]
                y_axis_label = c["y_axis_label"]
    except OSError:
        x_axis_label = None
        y_axis_label = None

    return x_axis_label, y_axis_label


def _make_legend(legend_title, is_classification, target_class_names, target_unit_legend=None, n_quantiles=None,
                 bins=None):
    if is_classification:
        legend = _make_legend_classification(legend_title=legend_title, target_class_names=target_class_names)
    else:
        legend = _make_legend_regression(legend_title=legend_title, n_quantiles=n_quantiles, bins=bins,
                                         target_unit_legend=target_unit_legend)

    return legend


def _make_legend_regression(legend_title, n_quantiles, bins, target_unit_legend):
    legend_1 = '''
    <span class="results-small-text"> <p align="center"> ''' + str(legend_title) + '''</p></span>
    <p align="center">
    <ul class="legend">
    '''
    # NOTE: this is ugly and it should be changed but it is not trivial  to automatize
    # (also the colors should be changed accordingly)
    if n_quantiles == 5:
        legend_2 = '''
        <li><span class="quintile_1"></span><div class="legend-small-text">''' + '[' + str(bins[0]) + str(
            target_unit_legend) + ', ' + str(bins[1]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_2"></span><div class="legend-small-text">''' + '[' + str(bins[1]) + str(
            target_unit_legend) + ', ' + str(bins[2]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_3"></span><div class="legend-small-text">''' + '[' + str(bins[2]) + str(
            target_unit_legend) + ', ' + str(bins[3]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_4"></span><div class="legend-small-text">''' + '[' + str(bins[3]) + str(
            target_unit_legend) + ', ' + str(bins[4]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_5"></span><div class="legend-small-text">''' + '[' + str(bins[4]) + str(
            target_unit_legend) + ', ' + str(bins[5]) + str(target_unit_legend) + ')' + '''</div> </li>
        </ul>
        </p>'''
    elif n_quantiles == 4:
        legend_2 = '''
        <li><span class="quintile_1"></span><div class="legend-small-text">''' + '[' + str(bins[0]) + str(
            target_unit_legend) + ', ' + str(bins[1]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_2"></span><div class="legend-small-text">''' + '[' + str(bins[1]) + str(
            target_unit_legend) + ', ' + str(bins[2]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_3"></span><div class="legend-small-text">''' + '[' + str(bins[2]) + str(
            target_unit_legend) + ', ' + str(bins[3]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_4"></span><div class="legend-small-text">''' + '[' + str(bins[3]) + str(
            target_unit_legend) + ', ' + str(bins[4]) + str(target_unit_legend) + ')' + '''</div> </li>
        </ul>
        </p>'''
    elif n_quantiles == 3:
        legend_2 = '''
        <li><span class="quintile_1"></span><div class="legend-small-text">''' + '[' + str(bins[0]) + str(
            target_unit_legend) + ', ' + str(bins[1]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_2"></span><div class="legend-small-text">''' + '[' + str(bins[1]) + str(
            target_unit_legend) + ', ' + str(bins[2]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_3"></span><div class="legend-small-text">''' + '[' + str(bins[2]) + str(
            target_unit_legend) + ', ' + str(bins[3]) + str(target_unit_legend) + ')' + '''</div> </li>
        </ul>
        </p>'''
    elif n_quantiles == 2:
        legend_2 = '''
        <li><span class="quintile_1"></span><div class="legend-small-text">''' + '[' + str(bins[0]) + str(
            target_unit_legend) + ', ' + str(bins[1]) + str(target_unit_legend) + ')' + '''</div> </li>
        <li><span class="quintile_2"></span><div class="legend-small-text">''' + '[' + str(bins[1]) + str(
            target_unit_legend) + ', ' + str(bins[2]) + str(target_unit_legend) + ')' + '''</div> </li>
        </ul>
        </p>'''
    elif n_quantiles == 1:
        legend_2 = '''
        <li><span class="quintile_1"></span><div class="legend-small-text">''' + '[' + str(bins[0]) + str(
            target_unit_legend) + ', ' + str(bins[1]) + str(target_unit_legend) + ')' + '''</div> </li>
        </ul>
        </p>'''
    else:
        legend_2 = ''

    legend = legend_1 + legend_2

    return legend


def _make_legend_classification(legend_title, target_class_names):
    legend_1 = '''<span class=\"results-small-text\"> <p align=\"center\"> {0}</p></span>'''.format(str(legend_title))
    legend_2_list = []
    for i in range(len(target_class_names)):
        legend_2_list.append(
            '''<li><span class="label_''' + str(i) + '''"></span><div class="legend-small-text">''' + str(
                target_class_names[i]) + '''</div> </li>''')

    legend_2_ = ''.join(legend_2_list)
    legend_2 = '''<p align=\"center\">\n <ul class=\"legend\">{0}</ul> </p>'''.format(legend_2_)
    legend = legend_1 + legend_2

    return legend


def copy_jsmol(configs, tmp_folder):
    """Copy jsmol folder to tmp folder.

    It is required because for security reasons jsmol needs to be in the same directory as the Viewer
    or below it in the directory tree.

    """
    # copy jsmol folder to tmp folder
    try:
        jsmol_folder = configs['html']['jsmol_folder']
        destination = os.path.join(tmp_folder, 'jsmol')
        copy_directory(jsmol_folder, destination)
    except OSError as err:
        logger.warning("{}".format(err))


def load_templates(template_to_load):
    tooltip_pred_target = """
                            <table class="nomad-tooltip">
                                        <tr>
                                            <th class="nomad-header-tooltip">System description
                                            <span style="font-size: 10px; color: #cccccc;">[$index]</span>
                                            </th>
                                        </tr>
                                    <tr>
                                        <td>
                                            <span class="small-text-tooltip"">Chemical formula: </span>
                                            <span class="small-text-tooltip"">@chemical_formula</span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td align="center">
                                            <p class="small-text-tooltip">Atomic structure preview </p>

                                        <img
                                            src="@imgs" height="150" alt="@imgs" width="150"
                                            style="float: center; margin: 15px 15px 15px 15px;"
                                            border="1"
                                        ></img>
                                        </td>
                                    </tr>
                                        <tr>
                                        <td align="center">
                                            <p class="small-text-tooltip"> (click to load an interactive 3D view below)</p>
                                        </td>
                                        </tr>

                                        <tr>
                                            <th class="nomad-header-tooltip">Predictions on this system </th>
                                        </tr>

                                    <tr>
                                        <td>
                                            <span class="small-text-tooltip">Ref. {0} = </span>
                                            <span class="small-text-tooltip">@target {1}  </span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <span class="small-text-tooltip"> Pred. {0} = </span>
                                            <span class="small-text-tooltip">@target_pred {1} </span>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <span class="small-text-tooltip"> Abs. error = </span>
                                            <span class="small-text-tooltip-error">@abs_error {1} </span>
                                        </td>
                                    </tr>
                                    <tr>
                                            <th class="nomad-header-tooltip"> More info </th>
                                        </tr>
                                    <tr>
                                        <td>
                                            <span class="small-text-tooltip">(x, y) = </span>
                                            <span class="small-text-tooltip">(@x, @y) </span>
                                        </td>
                                    </tr>

                            </table>

                                """

    tooltip_unsupervised = """
                <table class="nomad-tooltip">
                            <tr>
                                <th class="nomad-header-tooltip">System description
                                <span style="font-size: 10px; color: #cccccc;">[$index]</span>
                                </th>
                            </tr>
                        <tr>
                            <td>
                                <span class="small-text-tooltip"">Chemical formula: </span>
                                <span class="small-text-tooltip"">@chemical_formula</span>
                            </td>
                        </tr>
                        <tr>
                            <td align="center">
                                <p class="small-text-tooltip">Atomic structure preview </p>

                            <img
                                src="@imgs" height="150" alt="@imgs" width="150"
                                style="float: center; margin: 15px 15px 15px 15px;"
                                border="1"
                            ></img>
                            </td>
                        </tr>
                            <tr>
                            <td align="center">
                                <p class="small-text-tooltip"> (click to load an interactive 3D view below)</p>
                            </td>
                            </tr>


                        <tr>
                                <th class="nomad-header-tooltip"> More info </th>
                            </tr>
                        <tr>
                            <td>
                                <span class="small-text-tooltip">(x, y) = </span>
                                <span class="small-text-tooltip">(@x, @y) </span>
                            </td>
                        </tr>

                </table>
                    """

    html_viewer_instructions_with_jsmol = """
        </td>

            <td style="vertical-align: top;">

            <table class="instructions-table">
                <tr>
                     <td class="instructions-title-text">Instructions </td>
                </tr>
                <tr>
                    <td colspan=2 class="instructions-text">

                    On the left, we provide an <b><i>interactive</i></b> plot of the data-analytics results. <br>
                    A menu to turn on/off interactive functions is located on the left side of the plot (just below the pinwheel logo).
                    <br><br>

                    <span class="instructions-h1-text"> Basic features </span>

                    <ul>
                        <li> By <i>hovering</i> over a point in the plot, information regarding that system is displayed. </li>
                        <li> By <i>clicking</i> over a point, an interactive 3D visualization of the structure appears
                        in one of the bottom panels (alternating left and right panel at each click,
                        for comparing the last two selections). </li>

                    </ul>

                    <span class="instructions-h1-text"> Advanced features </span>

                    <ul>
                        <li> You can <i>zoom-in</i> on a selected area activating the <i>box zoom</i> function (2nd button from the top).
                        The full plot is still shown in the map on the right-side of this webpage, and a shaded area indicates where the selected area is in the plot. </li>
                        <li> You can modify the <i>aspect-ratio</i> activating the <i>resize</i> function (3rd button from the top),
                        and dragging the bottom-right corner of the plot.</li>
                    </ul>

                    </td>
                </tr>
            </table>

            </td>

                        <td style="height:100%">
                            <table style="height:100%">
                                <tr>
                                    <td align="center" style="width: 100%; height:320px; vertical-align: top">
                                        {{ div['Map'] }}
                                    </td>
                                </tr>
                                <tr>
                                    <td align="center" style=" vertical-align: top">
                                        <table id="clustering_info" align="center">
                                                <tr class='clickablea-row' data-href='url://www.google.com'>
                                                    <th colspan=3 class="selection"> Selection </th>
                                                </tr>

                                                """

    html_viewer_instructions = """
        </td>

            <td style="vertical-align: top;">

            <table class="instructions-table">
                <tr>
                     <td class="instructions-title-text">Instructions </td>
                </tr>
                <tr>
                    <td colspan=2 class="instructions-text">

                    On the left, we provide an <b><i>interactive</i></b> plot of the data-analytics results. <br>
                    A menu to turn on/off interactive functions is located on the left side of the plot (just below the pinwheel logo).
                    <br><br>

                    <span class="instructions-h1-text"> Basic features </span>

                    <ul>
                        <li> By <i>hovering</i> over a point in the plot, information regarding that system is displayed. </li>
                    </ul>

                    <span class="instructions-h1-text"> Advanced features </span>

                    <ul>
                        <li> You can <i>zoom-in</i> on a selected area activating the <i>box zoom</i> function (2nd button from the top).
                        The full plot is still shown in the map on the right-side of this webpage, and a shaded area indicates where the selected area is in the plot. </li>
                        <li> You can modify the <i>aspect-ratio</i> activating the <i>resize</i> function (3rd button from the top),
                        and dragging the bottom-right corner of the plot.</li>
                    </ul>

                    </td>
                </tr>
            </table>

            </td>

                        <td style="height:100%">
                            <table style="height:100%">
                                <tr>
                                    <td align="center" style="width: 100%; height:320px; vertical-align: top">
                                        {{ div['Map'] }}
                                    </td>
                                </tr>
                                                """

    html_recap_table_header_pred_target = """<tr>
                                        <th> Name </th>
                                        <th> Reference {0} [{1}] </th>
                                        <th> Predicted {0} [{1}] </th>
                                    </tr>"""

    html_recap_table_header_no_pred_target = """ <tr>
                                            <th> Name </th>
                                            <th> {0} [{1}] </th>
                                        </tr>"""

    html_recap_table_clear_selection = """
                                        </table>
                                      <INPUT type="button" value="Clear Selection" onclick="deleteRow('clustering_info')" />
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td align="center">
                            <table id="jsmol_table">
                            <tr>
                                <th>Name</th>
                                <th>Geometry File</th>
                            </tr>
                            <tr>
                                <td> <div id="chemical_formula0"> &nbsp; </div> </td>
                                <td> &nbsp; <a id="geo_link0"></a>  </td>
                            </tr>
                            <tr>
                                <td colspan=2 class="none">
                                    <div id="appdiv0"></div>
                                </td>
                            </tr>
                            </table>
                        </td>


                        <td align="center">
                            <table id="jsmol_table">
                            <tr>
                                <th>Name</th>
                                <th>Geometry File</th>
                            </tr>
                            <tr>
                                <td> <div id="chemical_formula1"> &nbsp; </div> </td>
                                <td> &nbsp; <a id="geo_link1"></a>  </td>
                            </tr>
                            <tr>
                                <td colspan=2 class="none">
                                    <div id="appdiv1"></div>
                                </td>
                            </tr>
                            </table>
                        </td>
                    </tr>
            <tr>
                <td colspan=2>
                    <table>
                        <tr>
                            <td style="width:10%"> &nbsp;  </td>
                            <td>
                                <span class="results-small-text"><br> {0} </span>
                            </td>
                            <td> &nbsp; </td>
                        </tr>
                    </table>
                </td>
            </tr>
                </table>

            </tr>
            <tr>
                <td> &nbsp; </td>
            </tr>

            </table>
        </body>
    </html>
    """

    show_outfile = """ </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    <tr>
                <td colspan=2>
                    <table>
                        <tr>
                            <td style="width:10%"> &nbsp;  </td>
                            <td>
                                <span class="results-small-text"><br> {0} </span>
                            </td>
                            <td> &nbsp; </td>
                        </tr>
                    </table>
                </td>
            </tr>
                </table>"""

    # these two templates (html_page_head_1 and html_page_head_2) were kept separate because
    # when I merged them, a strange error occured: KeyError: '\n            document'
    html_page_head_1 = '''<!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
            <script>
            document.title = "NOMAD viewer";
            </script>

                 <link rel="stylesheet" href="./jsmol/bokeh-0.11.1.min.css" type="text/css" />
                 <link rel="stylesheet" href="https://netdna.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.css">

            <link rel="stylesheet" type="text/css" href="{}'''

    html_page_head_2 = '''">

                <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.0/jquery.min.js"></script>
               <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.11.1.min.js"></script>
               <script>window.Bokeh || document.write('<script src="./jsmol/bokeh-0.11.1.min.js"></script>
               <script src="https://code.jquery.com/jquery-1.11.2.js"></script>
                    {{ js_resources }}
                    {{ css_resources }}
                    {{ script }}

            <script>

            function writeInfoApplet0(chemical_formula_, geo_file_) {
            document.getElementById("chemical_formula0").innerHTML = String(chemical_formula_);
            document.getElementById("geo_link0").innerHTML = "View";
            document.getElementById("geo_link0").href = String(geo_file_);
            document.getElementById("geo_link0").target = "_blank";
            };

            function writeInfoApplet1(chemical_formula_, geo_file_) {
            document.getElementById("chemical_formula1").innerHTML = String(chemical_formula_);
            document.getElementById("geo_link1").innerHTML = "View";
            document.getElementById("geo_link1").href = String(geo_file_);
            document.getElementById("geo_link1").target = "_blank";
            };'''

    html_template_viewer_2 = '''function deleteRow(tableID) {
                    var table = document.getElementById(tableID);
                    var rowCount = table.rows.length;

                    for(var i=2; i<rowCount; i++) {
                        var row = table.rows[i];
                        table.deleteRow(i);
                        rowCount--;
                        i--;
                        }
                    };

                </script>
    <style>



    .legend { list-style: none; }
    .legend li { float: left; margin-right: 10px; }
    .legend span { border: 1px solid #ccc; float: left; width: 12px; height: 12px; margin: 2px;
    }
    /* your colors */


    .legend .label_0 { background-color: #000000; }
    .legend .label_1 { background-color: #0072B2; }
    .legend .label_2 { background-color: #009E73; }
    .legend .label_3 { background-color: #E69F00; }
    .legend .label_4 { background-color: #CC79A7; }


    .legend .label_5 { background-color: #2196f3; }
    .legend .label_6 { background-color: #03a9f4; }
    .legend .label_7 { background-color: #00bcd4; }
    .legend .label_8 { background-color: #009688; }
    .legend .label_9 { background-color: #4caf50; }
    .legend .label_10 { background-color: #8bc34a; }
    .legend .label_11 { background-color: #cddc39; }
    .legend .label_12 { background-color: #ffeb3b; }
    .legend .label_13 { background-color: #ffc107; }
    .legend .label_14 { background-color: #ff9800; }
    .legend .label_15 { background-color: #ff5722; }
    .legend .label_16 { background-color: #795548; }
    .legend .label_17 { background-color: #9e9e9e; }
    .legend .label_18 { background-color: #607d8b; }
    .legend .label_19 { background-color: #b71c1c; }
    .legend .label_20 { background-color: #880e4f; }
    .legend .label_21 { background-color: #4a148c; }
    .legend .label_22 { background-color: #311b92; }
    .legend .label_23 { background-color: #1a237e; }
    .legend .label_24 { background-color: #0d47a1; }
    .legend .label_25 { background-color: #01579b; }
    .legend .label_26 { background-color: #006064; }
    .legend .label_27 { background-color: #004d40; }
    .legend .label_28 { background-color: #1b5e20; }
    .legend .label_29 { background-color: #33691e; }
    .legend .label_30 { background-color: #827717; }
    .legend .label_31 { background-color: #f57f17; }
    .legend .label_32 { background-color: #ff6f00; }
    .legend .label_33 { background-color: #e65100; }
    .legend .label_34 { background-color: #bf360c; }
    .legend .label_35 { background-color: #3e2723; }
    .legend .label_36 { background-color: #212121; }
    .legend .label_37 { background-color: #263238; }



    .legend .quintile_1 { background-color: #000000; }
    .legend .quintile_2 { background-color: #0072B2; }
    .legend .quintile_3 { background-color: #009E73; }
    .legend .quintile_4 { background-color: #E69F00; }
    .legend .quintile_5 { background-color: #CC79A7; }


     </style>

       </head><body id='fullwidth' class='fullwidth page-1'>
        <table style="width: 100%, border: 4">

            <tr>
                <table class="headerNOMAD">
                    <tr>
                        <td class="label">
                            <img id="nomad" src="https://nomad-coe.eu/uploads/nomad/images/NOMAD_Logo2.png" width="229" height="100" alt="NOMAD Logo" />
                        </td>
                        <td class="input">
                           <span class="header-large-text">Viewer<br></span>
                            <span class="header-small-text">The&nbsp;NOMAD&nbsp;Laboratory <br></span>
                            <span>&nbsp;</span>
                        </td>
                    </tr>
                </table>
            </tr>
            <tr>

                <table align="center" style="background-color: #F5F5F5">
                    <tr align="center">
                        <td style="vertical-align: top;">
                            {{ div['main_plot'] }}'''

    templates = dict(tooltip_pred_target=tooltip_pred_target, tooltip_unsupervised=tooltip_unsupervised,
                     html_viewer_instructions=html_viewer_instructions,
                     html_viewer_instructions_with_jsmol=html_viewer_instructions_with_jsmol,
                     html_recap_table_header_pred_target=html_recap_table_header_pred_target,
                     html_recap_table_header_no_pred_target=html_recap_table_header_no_pred_target,
                     html_recap_table_clear_selection=html_recap_table_clear_selection,
                     html_page_head_1=html_page_head_1, html_page_head_2=html_page_head_2,
                     html_template_viewer_2=html_template_viewer_2, show_outfile=show_outfile)

    return templates[template_to_load]


def load_js_scripts(script_to_load):
    load_jsmol_applet = """
            <script type="text/javascript" src="./jsmol/JSmol.min.js"></script>
            <script type="text/javascript">
            Jmol._isAsync = false;
            Jmol.getProfile() // records repeat calls to overridden or overloaded Java methods
            var jmolApplet0; // set up in HTML table, below
            var jmolApplet1; // set up in HTML table, below
            var chemical_formula;
            // use ?_USE=JAVA or _USE=SIGNED or _USE=HTML5
            jmol_isReady = function(applet) {
                Jmol._getElement(applet, "appletdiv").style.border="0px solid black"
             }

            var  Info = {
                width: 400,
                height: 300,
                debug: false,
                color: "#FFFFFF",
                //color: "#F0F0F0",
                zIndexBase: 20000,
                z:{monitorZIndex:100},
                serverURL: "./php/jsmol.php",
                use: "HTML5",
                jarPath: "./jsmol/java",    // this needs to point to where the j2s directory is.
                j2sPath: "./jsmol/j2s",     // this needs to point to where the java directory is.
                jarFile: "./jsmol/JmolApplet.jar",
                isSigned: false,
                disableJ2SLoadMonitor: true,
                disableInitialConsole: true,
                readyFunction: jmol_isReady,
                allowjavascript: true,
            }


                  $(document).ready(function() {

                  $("#appdiv0").html(Jmol.getAppletHtml("jmolApplet0", Info));
                  $("#appdiv1").html(Jmol.getAppletHtml("jmolApplet1", Info));
                  }
                  );


                  var lastPrompt=0;

                  </script>

                """

    write_summary_function_pred = '''
         function writeSummary(chemical_formula_, target_, target_pred_){
          //check if the user actually clicked on one point on the plot
          if (chemical_formula_ != null && target_pred_ != null){
              $("#clustering_info tbody").append(
              "<tr class='clickable-row' data-href='url://www.google.com'>"+
              "<td>" + String(chemical_formula_) + "</td>"+
              "<td>" + String(target_) + "</td>"+
              "<td>" + String(target_pred_) + "</td>"+
              "</tr>");
          }
          };
      '''

    write_summary_function_no_pred = '''
        function writeSummary(chemical_formula_, target_){
        //check if the user actually clicked on one point on the plot
        if (chemical_formula_ != null){
            $("#clustering_info tbody").append(
            "<tr class='clickable-row' data-href='url://www.google.com'>"+
            "<td>" + String(chemical_formula_) + "</td>"+
            "<td>" + String(target_) + "</td>"+
            "</tr>");
        }
        };
    '''
    js_zoom = """
            var data = source.get('data');

            //read from cb_obj the start and end of the selection
            var start = cb_obj.get('start');
            var end = cb_obj.get('end');

            // save the values in the data source
            data['%s'] = [start + (end - start) / 2];
            data['%s'] = [end - start];
            source.trigger('change');
        """

    js_load_jmol_1 = """
            // get data source from Callback args
            var data = source.get('data');

            // obtain the index of the point that was clicked
            // cb_obj contains information on the tool used
            var inds = cb_obj.get('selected')['1d'].indices;
            """
    js_load_jmol_2_pred = """
               //pick from the data source the corresponding file
               var geo_file = data['geo_file'][inds];
               var chemical_formula = data['chemical_formula'][inds];
               var target = data['target'][inds];
               var target_pred = data['target_pred'][inds];
               """

    js_load_jmol_2_no_pred = """
            //pick from the data source the corresponding file
            var geo_file = data['geo_file'][inds];
            var chemical_formula = data['chemical_formula'][inds];
            var target = data['target'][inds];
            """

    js_load_jmol_3 = """
            // load in which JSmol applet the structure should be loaded
            // it is an array because it is included in the ColumnDataSource which needs to be iterable
            var whichJSmol = data['whichJSmol'];


            // decide in which JSmol applet the structure should be loaded
            // swap the value between 0 and 1 to alternate the JSmol applet in which we should plot
            // only one value of the array is read (for convenience). It does not matter because the elements are all the same (either 0 or 1)
            // open the file in jsmol

            if (whichJSmol[inds] == 0) {
            var file= \"javascript:Jmol.script(jmolApplet0," + "'load "+ geo_file + "; rotate x 0; rotate y 0; rotate z 0; set bondTolerance 0.45; ')" ;
            //var file= \"javascript:Jmol.script(jmolApplet0," + "'load "+ geo_file + " {1 1 1}; rotate x 0; rotate y 0; rotate z 0; set bondTolerance 0.45; ')" ;
            //var file= \"javascript:Jmol.script(jmolApplet0," + "'load "+ geo_file + " {3 3 3}; rotate x 0; rotate y 0; rotate z 0; set bondTolerance 0.45; ')" ;
            //var file= \"javascript:Jmol.script(jmolApplet0," + "'load "+ geo_file + " {3 3 3}; rotate x 10; rotate y 12; rotate z 6; set bondTolerance 0.45; ')" ;
            location.href = file;
            // change all the values of the array
            for (var i = 0; i < whichJSmol.length; i++){
                   whichJSmol[i] = 1;
            }
            writeInfoApplet0(chemical_formula, geo_file);
            }
            else if (whichJSmol[inds] == 1) {
            //var file= \"javascript:Jmol.script(jmolApplet1," + "'load "+ geo_file + "; rotate x 10; rotate y 12; rotate z 6; set bondTolerance 0.45; ')" ;
            //var file= \"javascript:Jmol.script(jmolApplet1," + "'load "+ geo_file + " {3 3 3}; rotate x 0; rotate y 0; rotate z 0; set bondTolerance 0.45; ')" ;
            var file= \"javascript:Jmol.script(jmolApplet1," + "'load "+ geo_file + " {1 1 1}; rotate x 0; rotate y 0; rotate z 0; set bondTolerance 0.45; ')" ;
            location.href = file;
            for (var i = 0; i < whichJSmol.length; i++){
                whichJSmol[i] = 0;
            }
            writeInfoApplet1(chemical_formula, geo_file);
            }
            """

    js_load_jmol_4_pred = """writeSummary(chemical_formula, target, target_pred);"""

    js_load_jmol_4_no_pred = """writeSummary(chemical_formula, target);"""

    js_load_jmol_5 = """
            // save the modification in the ColumnDataSource to keep the information for the next user click
            data['whichJSmol'] = whichJSmol;
            source.trigger('change');

    """

    script_templates = dict(load_jsmol_applet=load_jsmol_applet, js_zoom=js_zoom,
                            write_summary_function_pred=write_summary_function_pred,
                            write_summary_function_no_pred=write_summary_function_no_pred,
                            js_load_jmol_1=js_load_jmol_1, js_load_jmol_2_no_pred=js_load_jmol_2_no_pred,
                            js_load_jmol_2_pred=js_load_jmol_2_pred, js_load_jmol_3=js_load_jmol_3,
                            js_load_jmol_4_pred=js_load_jmol_4_pred, js_load_jmol_4_no_pred=js_load_jmol_4_no_pred,
                            js_load_jmol_5=js_load_jmol_5)

    return script_templates[script_to_load]
