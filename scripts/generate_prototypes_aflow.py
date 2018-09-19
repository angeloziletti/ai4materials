#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "09/01/18"

import sys
import os.path

from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger
from mendeleev import element
import shutil

configs = set_configs()
logger = setup_logger(configs, level='INFO', display_configs=False)

# add / at the end
main_folder = '/u/ziang/calc_nomad_sim/prototypes_aflow_new/'

# directories
tmp_folder = os.path.abspath(os.path.normpath(os.path.join(main_folder, 'tmp')))

data_folder = '/u/ziang/calc_nomad_sim/prototypes_aflow_new/'

# to retrieve the data from gate
# scp -r ziang@gate.rzg.mpg.de://afs/ipp-garching.mpg.de/home/z/ziang/prototypes_aflow ./

z_min = 1
z_max = 90
el_list = [el.symbol for el in element(range(z_min, z_max + 1))]
a_list = [3., 5.]

protos = []

# bct139
# Note that In (A_tI2_139_a, In) and α–Pa (A_tI2_139_a, a-Pa) have the same AFLOW prototype label.
# in the A6 structure, c_a is near the fcc ratio of sqrt(2), 
# while in the Aa structure, c/a is near the bcc ratio of 1.
# Degeneracies
# c_a=sqrt(2) -> fcc
# c_a=1 -> bcc
#
# Defaults: 
# In (A6) Structure: A_tI2_139_a -> a: 4.6002; c_a: 1.07523585931
# a-Pa (Aa) Structure: A_tI2_139_a -> a: 3.932; c_a: 0.823499491353
# protos.append({"name": "A_tI2_139_a",
# "el": el_list,
# "a": np.linspace(4.0, 6.0, 1),
# "c/a": np.linspace(0.6, 2.0, 5)
# })


# bct141
# When c_a=sqrt(2) this structure is equivalent to diamond (A4).
# defaults:
# b-Sn (A5) Structure: A_tI4_141_a -> a: 5.8318; c_a: 0.545611989437
# protos.append({"name": "A_tI4_141_a",
# "el": el_list,
# "a": np.linspace(4.0, 6.0, 1),
# "c/a": np.linspace(0.4, 0.8, 5)
# })

# rh 166
# Note that b-Po (A_hR1_166_a, b-Po) and a-Hg (A_hR1_166_a, a-Hg) have the same AFLOW prototype label. 
# Degeneracies:
# c_a=sqrt(6) = 2.449489742783178 -> fcc
# c_a=sqrt(3/2) = 1.224744871391589 -> sc
# c_a=sqrt(3/8) = 0.6123724356957945 -> bcc
#
# Defaults:
# b-Po (Ai) Structure: A_hR1_166_a -> a: 5.07846; c_a: 0.968139947937
# a-Hg (A10) Structure: A_hR1_166_a -> a: 3.45741; c_a: 1.92728082582
# protos.append({"name": "A_hR1_166_a",
# "el": el_list,
# "a": np.linspace(4.0, 6.0, 1),
# "c/a": np.linspace(0.8, 2.2, 5)
# })

# hcp
# defaults:
# Hexagonal Close Packed (Mg, A3) Structure: A_hP2_194_c -> a: 3.2093; c_a: 1.62359393014
# a-La (A3') Structure: A_hP4_194_ac -> a: 3.77; c_a: 3.2175066313
protos.append(
    {"name": "A_hP2_194_c", "el": el_list, "a": a_list, "c/a": [1.4, 1.5, 1.6, 1.7, 1.8]})

# sc 
# defaults:
# a-Po (Ah) Structure: A_cP1_221_a -> a: 3.34
protos.append({"name": "A_cP1_221_a", "el": el_list, "a": a_list})

# fcc
protos.append({"name": "A_cF4_225_a", "el": el_list, "a": a_list})

# diamond
protos.append({"name": "A_cF8_227_a", "el": el_list, "a": a_list})

# bcc
protos.append({"name": "A_cI2_229_a", "el": el_list, "a": a_list})

# list obtained running generate_param_aflow
# c_a_list_po = [0.6123724356957945, 0.6276817465881893, 0.6429910574805843, 0.658300368372979, 0.673609679265374, 0.6889189901577688, 0.7042283010501635, 0.7195376119425585, 0.7348469228349533, 0.7501562337273483, 0.7654655446197431, 0.7807748555121379, 0.7960841664045328, 0.8113934772969277, 0.8267027881893226, 0.8420120990817174, 0.8573214099741122, 0.8726307208665072, 0.8879400317589019, 0.9032493426512969, 0.9185586535436917, 0.9338679644360866, 0.9491772753284815, 0.9644865862208764, 0.9797958971132712, 0.995105208005666, 1.0104145188980609, 1.0257238297904558, 1.0410331406828508, 1.0563424515752455, 1.0716517624676403, 1.0869610733600352, 1.1022703842524302, 1.117579695144825, 1.1328890060372199, 1.1481983169296146, 1.1635076278220093, 1.1788169387144043, 1.1941262496067993, 1.2094355604991942, 1.224744871391589, 1.2400541822839837, 1.2553634931763786, 1.2706728040687736, 1.2859821149611685, 1.3012914258535633, 1.316600736745958, 1.331910047638353, 1.347219358530748, 1.3625286694231427, 1.3778379803155376, 1.3931472912079323, 1.408456602100327, 1.4237659129927223, 1.439075223885117, 1.454384534777512, 1.4696938456699067, 1.4850031565623014, 1.5003124674546966, 1.5156217783470913, 1.5309310892394863, 1.546240400131881, 1.5615497110242758, 1.576859021916671, 1.5921683328090657, 1.6074776437014604, 1.6227869545938554, 1.63809626548625, 1.6534055763786453, 1.66871488727104, 1.6840241981634347, 1.6993335090558297, 1.7146428199482244, 1.7299521308406194, 1.7452614417330143, 1.760570752625409, 1.7758800635178038, 1.7911893744101988, 1.8064986853025937, 1.8218079961949887, 1.8371173070873834, 1.8524266179797786, 1.867735928872173, 1.883045239764568, 1.898354550656963, 1.9136638615493577, 1.9289731724417527, 1.9442824833341474, 1.9595917942265424, 1.9749011051189371, 1.990210416011332, 2.005519726903727, 2.0208290377961218, 2.0361383486885165, 2.0514476595809117, 2.0667569704733064, 2.0820662813657016, 2.097375592258096, 2.112684903150491, 2.127994214042886, 2.1433035249352805, 2.1586128358276757, 2.1739221467200704, 2.189231457612465, 2.2045407685048604, 2.219850079397255, 2.23515939028965, 2.2504687011820446, 2.2657780120744397, 2.2810873229668345, 2.296396633859229, 2.3117059447516244, 2.3270152556440187, 2.342324566536414, 2.3576338774288086, 2.3729431883212038, 2.3882524992135985, 2.4035618101059932, 2.4188711209983884, 2.434180431890783, 2.449489742783178, 2.464799053675573, 2.4801083645679673, 2.4954176754603625, 2.5107269863527573, 2.526036297245152, 2.541345608137547, 2.556654919029942, 2.571964229922337, 2.5872735408147314, 2.6025828517071266, 2.6178921625995217, 2.633201473491916, 2.648510784384311, 2.663820095276706, 2.6791294061691007, 2.694438717061496, 2.7097480279538906, 2.7250573388462853, 2.74036664973868, 2.7556759606310752, 2.7709852715234704, 2.7862945824158647, 2.80160389330826, 2.816913204200654, 2.8322225150930493, 2.8475318259854445, 2.8628411368778393, 2.878150447770234, 2.8934597586626287, 2.908769069555024, 2.9240783804474186, 2.9393876913398134, 2.9546970022322085, 2.970006313124603, 2.985315624016998, 3.000624934909393, 3.0159342458017875, 3.0312435566941827, 3.0465528675865774, 3.0618621784789726]

# face of crystals - figure 4b
# protos = []
# protos.append({
# "folder_name": "A_hR1_166_a_fig4",
# "name": "A_hR1_166_a",
# "el": ['Po'],
# "a": [5.0],
# "c/a": c_a_list_po
# })


for proto in protos:
    logger.info("Generating prototype: {}".format(proto["name"]))
    output_folder = os.path.abspath(os.path.normpath(os.path.join(data_folder, proto["name"])))

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    struct_id = 0
    for a in proto["a"]:
        for el in proto["el"]:
            if "c/a" in proto:
                for c_a in proto["c/a"]:
                    logger.info("el: {} a: {} c/a: {}".format(el, a, c_a))
                    filename = proto["name"] + '_' + str(a) + '_' + str(c_a) + '_' + el +'.aims'
                    filepath = os.path.abspath(os.path.normpath(os.path.join(output_folder, filename)))
                    os.system('/u/auro/bin/aflow --proto=' + str(proto["name"]) + ':' + str(el) + ' --params=' + str(
                        a) + ',' + str(c_a) + ' --aims > ' + filepath)
                    struct_id += 1
            else:
                logger.info("el: {} a: {}".format(el, a))
                filename = proto["name"] + '_' + str(a) + '_' + el + '.aims'
                filepath = os.path.abspath(os.path.normpath(os.path.join(output_folder, filename)))
                os.system('/u/auro/bin/aflow --proto=' + str(proto["name"]) + ':' + str(el) + ' --params=' + str(
                    a) + ' --aims > ' + filepath)
                struct_id += 1

sys.exit(1)
