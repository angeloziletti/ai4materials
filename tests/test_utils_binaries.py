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

import unittest
from ai4materials.utils.utils_data_retrieval import read_ase_db
from ai4materials.utils.utils_binaries import get_energy_diff_by_spacegroup
from ai4materials.utils.utils_config import get_data_filename

@unittest.skip("temporarily disabled")
class TestUtilsBinaries(unittest.TestCase):
    def setUp(self):
        ase_db_file_binaries = get_data_filename('data/db_ase/binaries_ghiringhelli2015.json')

        self.ase_atoms_binaries = read_ase_db(db_path=ase_db_file_binaries)

    def test_get_energy_diff_by_spacegroup(self):
        equiv_spgroups = [(225, 221), (216, 227)]
        dict_delta_e = get_energy_diff_by_spacegroup(self.ase_atoms_binaries, target='energy_total',
                                                     equiv_spgroups=equiv_spgroups)

        dict_delta_e_correct = dict(SeZn=4.2159179660687287e-20, InSb=1.2506570609555687e-20,
                                    AgCl=-6.8568799955090609e-21, SZn=4.4190167090138643e-20, BN=2.7430550221687302e-19,
                                    GaSb=2.4773702178144287e-20, BrRb=-2.6246943074069208e-20,
                                    BaTe=-6.0143598233041078e-20, BeSe=7.9298202208796193e-20,
                                    MgS=-1.3890792272791661e-20, AsB=1.4018696180162358e-19, AlAs=3.416831556361176e-20,
                                    BP=1.632978757533666e-19, TeZn=3.9253535511316122e-20, MgSe=-8.8603255975880209e-21,
                                    ClLi=-6.149391549293258e-21, FK=-2.3456843288794608e-20,
                                    BrLi=-5.2465217764284492e-21, BSb=9.3062289002001379e-20,
                                    ClRb=-2.5715504707788539e-20, GeSn=1.3083912918128912e-20,
                                    CsI=-2.6017342042091155e-20, CaTe=-5.6149286826500206e-20,
                                    ClK=-2.6349506219210773e-20, Sn2=2.7179163244033424e-21,
                                    BrCs=-2.4972695386489382e-20, CsF=-1.734569615637085e-20,
                                    BrCu=2.442400384019518e-20, CaSe=-5.7806170659176063e-20,
                                    AgF=-2.4634695420313482e-20, MgTe=-7.3560522736479063e-22,
                                    FLi=-9.5310792412059186e-21, CuF=-2.7272687327072279e-21,
                                    FNa=-2.3357835066436331e-20, C2=4.2114873809101575e-19, BaO=-1.4900011177054134e-20,
                                    AgBr=-4.8118839046830307e-21, MgO=-3.721451404088126e-20,
                                    FRb=-2.1724838814450727e-20, AlN=1.1687730189874494e-20, Si2=4.4727296163305501e-20,
                                    SiSn=2.1646816357014748e-20, OSr=-3.5297012817210187e-20,
                                    ClNa=-2.1307665354820141e-20, AsIn=2.14767895373523e-20, OZn=1.633710321777262e-20,
                                    CGe=1.3000748379902827e-19, CdO=-1.348413629348854e-20, InP=2.8709930119109753e-20,
                                    SSr=-5.9029656118592692e-20, InN=2.4628706405675984e-20,
                                    BaSe=-5.5025977545738119e-20, BrK=-2.6624325013472597e-20,
                                    BeTe=7.5075740576200973e-20, CdS=1.1643465692630618e-20,
                                    CdTe=1.8351256432814928e-20, GeSi=4.217091895225337e-20, GaP=5.5876198809236085e-20,
                                    CdSe=1.3389702567051265e-20, INa=-1.8399111846593045e-20,
                                    AlP=3.5080994271602511e-20, BeO=1.1084460139894976e-19, AsGa=4.3944144343852047e-20,
                                    Ge2=3.218012279776111e-20, SeSr=-6.0003270319365202e-20, CSi=1.071894196156348e-19,
                                    BaS=-5.1231589897332471e-20, AgI=5.9161045280535275e-21, GaN=6.9445584247860156e-20,
                                    CaS=-5.9141658617526103e-20, AlSb=2.5133142314028706e-20,
                                    IK=-2.6762621853286689e-20, ILi=-3.4704646494924847e-21,
                                    ClCs=-2.4088110334584613e-20, CaO=-4.2492775596486839e-20,
                                    CuI=3.2792483878850995e-20, CSn=7.2664795347721018e-20, BeS=8.1122637897805144e-20,
                                    IRb=-2.6788624696652254e-20, BrNa=-2.0256115610545176e-20,
                                    SrTe=-6.0769715445352658e-20, ClCu=2.5035406212316063e-20)

        self.assertDictEqual(dict_delta_e, dict_delta_e_correct)
        self.assertIsInstance(dict_delta_e, dict)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtilsBinaries)
    unittest.TextTestRunner(verbosity=2).run(suite)
