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
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/08/17"

from ai4materials.visualization.viewer import Viewer
from ai4materials.utils.utils_config import set_configs
import pandas as pd
from ai4materials.utils.utils_data_retrieval import read_ase_db
from ai4materials.visualization.viewer import read_control_file
from ai4materials.utils.utils_config import get_data_filename
import unittest
import six
import shutil
import tempfile
import webbrowser


class TestViewer(unittest.TestCase):
    def setUp(self):
        # create a temporary directories
        self.main_folder = tempfile.mkdtemp()
        configs = set_configs(main_folder=self.main_folder)
        self.configs = configs

        ase_db_file_binaries = get_data_filename('data/db_ase/binaries_lowest_energy_ghiringhelli2015.json')
        results_binaries_lasso = get_data_filename('data/viewer_files/l1_l0_dim2_for_viewer.csv')
        results_metal_non_metal = get_data_filename('data/viewer_files/tutorial_metal_non_metal_2017.csv')
        results_topological_ins = get_data_filename('data/viewer_files/tutorial_topological_insulators_2017.csv')
        control_file_binaries = get_data_filename('data/viewer_files/binaries_control.json')

        self.ase_atoms_binaries = read_ase_db(db_path=ase_db_file_binaries)
        self.results_binaries_lasso = results_binaries_lasso
        self.results_metal_non_metal = results_metal_non_metal
        self.control_file_binaries = control_file_binaries
        self.results_topological_ins = results_topological_ins

    def test_bulk_with_target(self):
        # test Viewer for three-dimensional periodic structures with target
        viewer = Viewer(configs=self.configs)

        df_viewer = pd.read_csv(self.results_binaries_lasso)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']
        target = df_viewer['y_true']
        target_pred = df_viewer['y_pred']

        file_html_link, file_html_name = viewer.plot_with_structures(x=x, y=y, target=target, target_pred=target_pred,
                                                                     ase_atoms_list=self.ase_atoms_binaries,
                                                                     target_unit='eV', is_classification=False,
                                                                     tmp_folder=self.configs['io']['tmp_folder'])

        # if you want to open the webpage to visually inspect it  # webbrowser.open(file_html_name)

    def test_bulk_no_target_pred(self):
        # test Viewer for three-dimensional periodic structures without target_pred and without target
        viewer = Viewer(configs=self.configs)

        df_viewer = pd.read_csv(self.results_binaries_lasso)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']
        target = df_viewer['y_true']

        file_html_link, file_html_name = viewer.plot_with_structures(x=x, y=y, target=target,
                                                                     ase_atoms_list=self.ase_atoms_binaries,
                                                                     target_unit='eV', is_classification=False,
                                                                     tmp_folder=self.configs['io']['tmp_folder'])

        # if you want to open the webpage to visually inspect it  # webbrowser.open(file_html_name)

    def test_bulk_no_target(self):
        # test Viewer for three-dimensional periodic structures without target_pred and without target
        viewer = Viewer(configs=self.configs)

        df_viewer = pd.read_csv(self.results_binaries_lasso)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']

        try:
            viewer.plot_with_structures(x=x, y=y, ase_atoms_list=self.ase_atoms_binaries, target_unit='eV',
                                        is_classification=False, tmp_folder=self.configs['io']['tmp_folder'])
        except ValueError as err:
            print("{0}".format(err))

    def test_load_control_file(self):
        # read x and y axis labels from control file
        x_axis_label, y_axis_label = read_control_file(self.control_file_binaries)

        self.assertIsInstance(x_axis_label, six.string_types)
        self.assertIsInstance(y_axis_label, six.string_types)

    def test_tutorial_binaries_regression(self):

        df_viewer = pd.read_csv(self.results_binaries_lasso)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']
        target = df_viewer['y_true']
        target_pred = df_viewer['y_pred']

        legend_title = 'Reference E(RS)-E(ZB)'
        target_name = 'E(RS)-E(ZB)'
        plot_title = 'SISSO(L0) structure map'

        viewer = Viewer(configs=self.configs)

        # read x and y axis labels from control file
        x_axis_label, y_axis_label = read_control_file(self.control_file_binaries)

        file_html_link, file_html_name = viewer.plot_with_structures(x=x, y=y, target=target, target_pred=target_pred,
                                                                     ase_atoms_list=self.ase_atoms_binaries,
                                                                     target_unit='eV', target_name=target_name,
                                                                     legend_title=legend_title, is_classification=False,
                                                                     x_axis_label=x_axis_label,
                                                                     y_axis_label=y_axis_label, plot_title=plot_title,
                                                                     tmp_folder=self.configs['io']['tmp_folder'])

        # import webbrowser  # webbrowser.open(file_html_name)

    def test_tutorial_metal_non_metal(self):

        df_viewer = pd.read_csv(self.results_metal_non_metal)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']
        target = df_viewer['target'].tolist()

        legend_title = 'Metal/non metal class'
        target_name = 'Metal/non metal class'
        plot_title = 'SISSO(L0) structure map'

        x_axis_label = '(IE(B))^2*(IPF/Xp(A))'
        y_axis_label = '(IPF/Xp(A))*(Xp(B)/x(B))'

        viewer = Viewer(configs=self.configs)

        df_tooltip = df_viewer[['compound', 'crystal structure type', 'target', 'coord_0', 'coord_1']]

        file_html_link, file_html_name = viewer.plot(x=x, y=y, target=target,
                                                     df_tooltip=df_tooltip, target_unit='eV', target_name=target_name,
                                                     legend_title=legend_title, show_convex_hull=True,
                                                     is_classification=True, x_axis_label=x_axis_label,
                                                     y_axis_label=y_axis_label, plot_title=plot_title,
                                                     tmp_folder=self.configs['io']['tmp_folder'])

        # import webbrowser
        # webbrowser.open(file_html_name)

    def test_tutorial_topological_insulators(self):

        df_viewer = pd.read_csv(self.results_topological_ins)
        x = df_viewer['coord_0']
        y = df_viewer['coord_1']
        target = df_viewer['target'].tolist()

        legend_title = 'QSH/trivial insulators'
        target_name = 'QSH/trivial insulators'
        plot_title = 'SISSO(L0) structure map'

        x_axis_label = '(IP(B)/EA(B))*(Z(B)+Z(A))'
        y_axis_label = '(IP(B)-EA(C))*(Z(C)/rs(A))'

        viewer = Viewer(configs=self.configs)

        df_tooltip = df_viewer[['compound', 'crystal structure type', 'target', 'coord_0', 'coord_1']]

        file_html_link, file_html_name = viewer.plot(x=x, y=y, target=target,
                                                     df_tooltip=df_tooltip, target_unit='eV', target_name=target_name,
                                                     legend_title=legend_title, show_convex_hull=True,
                                                     is_classification=True, x_axis_label=x_axis_label,
                                                     y_axis_label=y_axis_label, plot_title=plot_title,
                                                     tmp_folder=self.configs['io']['tmp_folder'])

        # import webbrowser
        # webbrowser.open(file_html_name)

    def tearDown(self):
        shutil.rmtree(self.main_folder)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestViewer)
    unittest.TextTestRunner(verbosity=2).run(suite)
