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

    import matplotlib.pyplot as plt
    from ai4materials.utils.utils_config import set_configs
    from ai4materials.utils.utils_config import setup_logger
    from ai4materials.utils.utils_data_retrieval import read_ase_db
    from ai4materials.wrappers import load_descriptor
    from ai4materials.wrappers import calc_model
    from ai4materials.wrappers import calc_descriptor
    from ai4materials.descriptors.atomic_features import AtomicFeatures
    from ai4materials.descriptors.atomic_features import get_table_atomic_features
    from ai4materials.utils.utils_config import get_data_filename
    from ai4materials.visualization.viewer import read_control_file
    import numpy as np
    import pandas as pd

    # modify this path if you want to save the calculation results in another location
    configs = set_configs(main_folder='./l1_l0_example')
    logger = setup_logger(configs, level='INFO')

    # setup folder and files
    lookup_file = os.path.join(configs['io']['main_folder'], 'lookup.dat')
    materials_map_plot_file = os.path.join(configs['io']['main_folder'], 'binaries_l1_l0_map_prl2015.png')

    # define descriptor - atomic features in this case
    kwargs = {'energy_unit': 'eV', 'length_unit': 'angstrom'}
    descriptor = AtomicFeatures(configs=configs, **kwargs)

    # =============================================================================
    # Descriptor calculation
    # =============================================================================

    desc_file_name = 'atomic_features_binaries'
    ase_db_file = get_data_filename('data/db_ase/binaries_lowest_energy_ghiringhelli2015.json')
    ase_atoms_list = read_ase_db(db_path=ase_db_file)

    selected_feature_list = ['atomic_ionization_potential', 'atomic_electron_affinity', 'atomic_rs_max',
                             'atomic_rp_max', 'atomic_rd_max']
    allowed_operations = ['+', '-', '/', '|-|', 'exp', '^2']

    desc_file_path = calc_descriptor(descriptor=descriptor, configs=configs, ase_atoms_list=ase_atoms_list,
                                     desc_file='lasso_l0_binaries_example.tar.gz',
                                     format_geometry='aims',
                                     selected_feature_list=selected_feature_list,
                                     nb_jobs=-1)

    # load descriptor
    target_list, structure_list = load_descriptor(desc_files=desc_file_path, configs=configs)
    df_atomic_features = get_table_atomic_features(structure_list)

    # =============================================================================
    # Model calculation
    # =============================================================================

    chemical_formulas = [structure.get_chemical_formula(mode='hill') for structure in structure_list]
    df_atomic_features['chemical_formula'] = chemical_formulas
    df_atomic_features = df_atomic_features.sort_values(by='chemical_formula').reset_index(drop=True)

    # target values to predict
    dict_delta_e = dict(SeZn=0.2631369195046646, BaTe=-0.37538683850924387, BN=1.7120803923951688,
                        CGe=0.8114429425515818, GaP=0.3487518245522925, MgS=-0.08669951164989079,
                        GaN=0.4334452723999156, AlAs=0.21326186549251072, BP=1.019225239514441, FK=-0.14640610974868423,
                        BrLi=-0.03274621540254649, BSb=0.5808491589999847, CaTe=-0.3504563060008138,
                        ClK=-0.16446069285018655, BrCs=-0.1558673149861294, BrCu=0.15244265149855352,
                        ILi=-0.021660938008450818, CuF=-0.01702227364862989, FNa=-0.14578814899027592,
                        C2=2.6286038411199026, AgBr=-0.030033419005850936, CuI=0.20467459898973175,
                        GaSb=0.15462529698986593, ClLi=-0.03838148564873346, AsIn=0.13404758548892423,
                        OZn=0.10196818460305757, MgO=-0.2322747421651549, InP=0.17919330099729866,
                        Ge2=0.20085254149716641, InN=0.15372030450150198, CSn=0.45353800899655555,
                        CdTe=0.11453954098812649, TeZn=0.24500131400199776, MgTe=-0.004591286999846332,
                        BaS=-0.3197624539995756, CaSe=-0.36079776214906895, FRb=-0.1355957874033439,
                        BeO=0.6918376303948839, AsB=0.8749782510022386, CaS=-0.36913322290101264,
                        CaO=-0.2652190617003161, BaO=-0.09299856100784433, AlSb=0.15686874600534004,
                        SrTe=-0.3792947550252322, BeS=0.5063277134499351, InSb=0.0780598790169251,
                        SZn=0.27581334679854935, OSr=-0.2203066401004525, BrRb=-0.1638205440075271,
                        BeSe=0.4949404808020511, ClRb=-0.16050356640655905, BrNa=-0.1264287376032476,
                        MgSe=-0.05530180620975655, GeSn=0.08166336650886348, GeSi=0.2632101904042582,
                        CsF=-0.10826332699038382, CdSe=0.08357195550137826, FLi=-0.059488321434879074,
                        AlN=0.07294907877519896, Si2=0.2791658430004932, SiSn=0.13510880949563495,
                        ClNa=-0.13299199530041886, CdO=-0.0841613645001312, SSr=-0.36843415824218,
                        IK=-0.16703915799644553, BaSe=-0.3434451604764059, BrK=-0.1661759769597461,
                        BeTe=0.4685859464949282, CdS=0.07267280149604124, CsI=-0.16238748698990838,
                        INa=-0.11483823100687315, AlP=0.2189583583002711, AsGa=0.27427779349540243,
                        SeSr=-0.3745109805057823, CSi=0.669023778644634, AgCl=-0.04279728149250233,
                        AgI=0.03692542249419624, AgF=-0.15375768499313544, ClCs=-0.1503461689991465,
                        Sn2=0.016963900503544026, ClCu=0.15625872520000064, IRb=-0.16720145498980848)

    df_atomic_features['target'] = df_atomic_features['chemical_formula'].map(dict_delta_e)
    target = np.asarray(df_atomic_features['target'].values.astype(float))

    cols_to_drop = ['chemical_formula', 'target', 'ordered_chemical_symbols']

    # use the l1-l0 method proposed in Ghiringhelli et al. (2015)
    calc_model(method='l1_l0', df_features=df_atomic_features, cols_to_drop=cols_to_drop,
               target=target, max_dim=2, allowed_operations=allowed_operations,
               tmp_folder=configs['io']['tmp_folder'], results_folder=configs['io']['results_folder'],
               lookup_file=lookup_file, control_file=configs['io']['control_file'], energy_unit='eV',
               length_unit='angstrom')

    # read the results for the two-dimensional descriptor
    viewer_filename = 'l1_l0_dim1_for_viewer.csv'
    viewer_filepath = os.path.join(configs['io']['results_folder'], viewer_filename)
    df_viewer = pd.read_csv(viewer_filepath)
    x_axis_label, y_axis_label = read_control_file(configs['io']['control_file'])

    # plot the results for the two-dimensional descriptor
    fig, ax = plt.subplots()
    x = df_viewer['coord_0']
    y = df_viewer['coord_1']
    color = df_viewer['y_true']
    chemical_formula = df_viewer['chemical_formula']
    cm = plt.cm.get_cmap('rainbow')
    sc = plt.scatter(x, y, c=color, cmap=cm)

    # annotate the points
    for i, txt in enumerate(chemical_formula):
        ax.annotate(txt, (x[i], y[i]),  size=4)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    cbar = plt.colorbar(sc)
    cbar.set_label('Reference E(RS)-E(ZB)', rotation=90)
    plt.title("l1/l0 structure map for binary compounds\n ")
    plt.subplots_adjust(bottom=0.2)
    plt.figtext(0.5, 0.02, "Compare with Fig. 2 in Ghiringhelli et al., Phys. Rev. Lett 114 (10), 105503 (2015)",
                horizontalalignment='center', style='italic')

    plt.savefig(materials_map_plot_file, dpi=300)

