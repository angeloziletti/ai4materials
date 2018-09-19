from __future__ import absolute_import
from __future__ import division

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/10/16"

import sys, os, os.path

base_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.normpath(os.path.join(base_dir,"../../../python-common/common/python"))
nomad_sim_dir = os.path.normpath(os.path.join(base_dir,"../python-modules/"))
atomic_data_dir = os.path.normpath(os.path.join(base_dir, '../../atomic-data')) 

if not common_dir in sys.path:
    sys.path.insert(0, common_dir) 
    sys.path.insert(0, nomad_sim_dir) 
    sys.path.insert(0, atomic_data_dir) 
    
import os
from scipy.misc import imread
from PIL import Image
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def plot_with_symmetric_colormap(img_array, outfile, img_folder, show=True,
    min_scale_value=None, max_scale_value=None, nb_ticks=8):
    # http://stackoverflow.com/questions/23994020/colorplot-that-distinguishes-between-positive-and-negative-values
    # define the colormap
    cmap = plt.get_cmap('PuOr')    
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize and forcing 0 to be part of the colorbar
    if min_scale_value is not None and max_scale_value is not None:
        bounds = np.arange(min_scale_value, max_scale_value, .1)

    else:
        bounds = np.arange(np.min(img_array),np.max(img_array),.5)
    
#    print bounds
    # find indices where elements (zero in our case) should be inserted to maintain order
    idx = np.searchsorted(bounds, 0)
    bounds = np.insert(bounds, idx, 0)
    
    norm = BoundaryNorm(bounds, cmap.N)
    # to avoid whitespaces when saving
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.imshow(img_array, norm=norm, cmap=cmap)
    
    if min_scale_value is not None and max_scale_value is not None:
        ticks_spacing = int((max_scale_value - min_scale_value)/nb_ticks)
        print(ticks_spacing)
        ticks = range(int(min_scale_value), int(max_scale_value)+1, ticks_spacing)
        print(ticks)
        plt.colorbar(ticks=ticks)
    else:
       plt.colorbar()

    outfile = os.path.abspath(os.path.normpath(os.path.join(img_folder, outfile)))
    plt.savefig(outfile, dpi=600)

    if show:
        plt.show()

    plt.clf()

    
def main():
    
    img_folder = '/home/ziletti/Documents/writing/face_of_crystals/2nd_version/images/descriptor_img/diffraction_images_subtraction'
    
    img_file_pr = 'spgroup221_256_pristine.png'
    img_file_dsp = 'spgroup221_256_disp008.png'
    img_file_vc = 'spgroup221_256_vac25.png'

    img_file_diff_pr_dsp = 'spgroup221_256_pristine_disp'
    img_file_diff_pr_vc = 'spgroup221_256_pristine_vac'
    
    img_file_pr = os.path.abspath(os.path.normpath(os.path.join(img_folder, img_file_pr)))
    img_file_dsp = os.path.abspath(os.path.normpath(os.path.join(img_folder, img_file_dsp)))
    img_file_vc = os.path.abspath(os.path.normpath(os.path.join(img_folder, img_file_vc)))
    
    img_pr = imread(img_file_pr).astype(float)
    img_dsp = imread(img_file_dsp).astype(float)
    img_vc = imread(img_file_vc).astype(float)

    # check if all images have same shape    
    assert np.asarray(img_pr.shape).all() == np.asarray(img_dsp.shape).all()
    assert np.asarray(img_pr.shape).all() == np.asarray(img_vc.shape).all()
    assert np.asarray(img_dsp.shape).all() == np.asarray(img_vc.shape).all()

    width, height, channels = img_pr.shape
    logger.info("Images have shape: {}".format(img_pr.shape))

    if channels == 3 or channels ==4:
        logger.info("Images are RGB.")
    elif channels == 1:
        logger.info("Images are Greyscale")
    
    logger.info("Max value in the pristine image (to check normalization): {}".format(img_pr.max()))
    logger.info("Max value in the random disp image (to check normalization): {}".format(img_dsp.max()))
    logger.info("Max value in the vacancies image (to check normalization): {}".format(img_vc.max()))
    
    diff_pr_dsp = img_dsp - img_pr 
    diff_pr_vc = img_vc - img_pr 
    
    diff_pr_dsp_min_perc = diff_pr_dsp.min()/img_pr.max()*100.0
    diff_pr_dsp_max_perc = diff_pr_dsp.max()/img_pr.max()*100.0
    # set boundaries as the maximum 
#    boundaries_pr_dsp = max(abs(diff_pr_dsp_min_perc), abs(diff_pr_dsp_max_perc))
    # set boundaries as the average 
    boundaries_pr_dsp = (abs(diff_pr_dsp_min_perc) + abs(diff_pr_dsp_max_perc))/2.0
    
    diff_pr_vac_min_perc = diff_pr_vc.min()/img_pr.max()*100.0
    diff_pr_vac_max_perc = diff_pr_vc.max()/img_pr.max()*100.0
    boundaries_pr_vac = (abs(diff_pr_vac_min_perc) + abs(diff_pr_vac_max_perc))/2.0
    
    logger.info("Random disp vs pristine min-max differences.")
    logger.info("Min: {0}[abs]; {1}%".format(diff_pr_dsp.min(), diff_pr_dsp_min_perc))
    logger.info("Max: {0}[abs]; {1}%".format(diff_pr_dsp.max(), diff_pr_dsp_max_perc))

    logger.info("Vacancies vs pristine min-max differences.")
    logger.info("Min: {0}[abs]; {1}%".format(diff_pr_vc.min(), diff_pr_vac_min_perc))
    logger.info("Max: {0}[abs]; {1}%".format(diff_pr_vc.max(), diff_pr_vac_max_perc))
            
    # get percentage difference
    diff_pr_dsp = np.divide(diff_pr_dsp, img_pr.max())*100.0
    diff_pr_vc = np.divide(diff_pr_vc, img_pr.max())*100.0

    # works only for greyscale and rgb images (w/o transparencies)
    # NB: adding small numbers is a hack to have 0 included in the interval
    # the small numbers they are just chosen to have 0 as tick
    for idx_ch in range(channels):
        # vacancies
        plot_with_symmetric_colormap(diff_pr_dsp[:,:,idx_ch], 
            img_file_diff_pr_dsp + "_heat_map_" + "ch" + str(idx_ch) + ".png", 
            img_folder, show=False,
            min_scale_value=-boundaries_pr_dsp+0.2, max_scale_value=boundaries_pr_dsp)
        # displacement
        plot_with_symmetric_colormap(diff_pr_vc[:,:,idx_ch], 
            img_file_diff_pr_vc + "_heat_map_" + "ch" + str(idx_ch) + ".png", 
            img_folder, show=False,
            min_scale_value=-boundaries_pr_vac+0.5, max_scale_value=boundaries_pr_vac)


if __name__ == "__main__":
    main()