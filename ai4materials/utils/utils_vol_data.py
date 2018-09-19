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
import numpy as np
import math
from scipy.interpolate import griddata
from scipy import ndimage

logger = logging.getLogger('ai4materials')


def get_slice_volume_indices(vol_data, min_r, max_r, dr=1.0, phi_bins=100, theta_bins=50):
    """Given 3d volume return the indices of points belonging to the specified concentric shells.

    The volume is should be centered according to its center of mass to have centered concentric shells.
    In our case we do not have to do that because the diffraction intensity obtained with the Fourier Transform
    is centered by definition. For use reference, we nevertheless calculate the center of mass within the function.

    Parameters:

    vol_data:
        Numpy 3D array containing the volumetric data to be sliced. In our case, this is the three-dimensional
        diffraction intensity.

    theta_bins: int, optional (default=50)
        Bins to be used for the theta angle of the spherical coordinates.

    phi_bins: int, optional (default=100)
        Bins to be used for the phi angle of the spherical coordinates.

    Returns: list
        List of length = (max_r - min_r)/dr; the length corresponds to the number of concentric shells considered.
        Each element in the list - representing a concentric shell - contains a list of 3 dimensional tuples,
        with the indices of the volume elements which belong to the given concentric shell.
        For example, let us assume the output of the function is stored in variable `xyz_r`.
        `xyz[0]` gives a list of tuples corresponding to the points in the first concentric shell.
        If `xyz[0][0] = (82, 97, 119)`, this means that the element of the volumetric shape with index
        (82, 97, 119) belong to the first shell.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    px, py, pz = vol_data.shape
    center_of_mass = ndimage.measurements.center_of_mass(vol_data)
    logger.debug("Volumetric data dimensions: {} {} {}".format(px, py, pz))
    logger.debug("Center of mass: {}".format(center_of_mass))
    x0 = center_of_mass[0]
    y0 = center_of_mass[1]
    z0 = center_of_mass[2]

    max_x = px - 1
    max_y = py - 1
    max_z = pz - 1
    assert max_x == max_y and max_x == max_z and max_y == max_z

    r_bins = int((max_r - min_r) / dr) + 1

    # create the grids
    r = np.linspace(min_r, max_r, r_bins)
    phi = np.linspace(0, 2 * np.pi, phi_bins)
    theta = np.linspace(0, np.pi, theta_bins)

    # spherical coordinates
    x = 1 * np.outer(np.cos(phi), np.sin(theta))
    y = 1 * np.outer(np.sin(phi), np.sin(theta))
    z = 1 * np.outer(np.ones(np.size(phi)), np.cos(theta))

    x_r = np.outer(r, x)
    y_r = np.outer(r, y)
    z_r = np.outer(r, z)

    xyz_r = []
    # get x,y,z indices of the points belonging to each spherical shell
    for i, r_i in enumerate(r):
        x_sh = x_r[i, :].flatten()
        y_sh = y_r[i, :].flatten()
        z_sh = z_r[i, :].flatten()

        coord = zip(np.rint(x_sh + x0).astype(int), np.rint(y_sh + y0).astype(int), np.rint(z_sh + z0).astype(int))
        xyz = list(set([xyz for xyz in coord if 0 <= xyz[0] < max_x and 0 <= xyz[1] < max_y and 0 <= xyz[2] < max_z]))
        xyz_r.append(xyz)

    return xyz_r


def _append_spherical_np(xyz):
    """Fast conversion from cartesian to spherical coordinates:

    We use the following definition for spherical coordinates
    (https://en.wikipedia.org/wiki/Spherical_coordinate_system):

    radius = sqrt(x^2+y^2+z^2) = math.sqrt((item[0]-x0)**2 + (item[1]-y0)**2 + (item[2]-z0)**2)
    theta_sel = arccos(z/r) = math.degrees(math.acos((item[2]-z0)/radius))
    phi_sel = arctan(y/z) = math.degrees(math.atan2(item[1]-y0, item[0]-x0))

    """

    # xyz[:,0] = ptsnew[:,0] = x
    # xyz[:,1] = ptsnew[:,1] = y
    # xyz[:,2] = ptsnew[:,2] = z
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2

    # ptsnew[:,3] --> r
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)

    # ptsnew[:,4] --> theta
    ptsnew[:, 4] = np.degrees(np.arccos(xyz[:, 2] / ptsnew[:, 3]))

    # ptsnew[:,5] --> phi
    ptsnew[:, 5] = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))

    return ptsnew


def get_shells_from_indices(xyz_r, vol_data):
    """Obtain concentric shells from volumetric data.

    The starting point are an array containing the volumetric data and a list of indices which assign points
    of the volume to the corresponding concentric shell. Using these indices, we perform two operations: \n
    1) extract the concentric shells in the volumetric space \n
    2) transform the concentric shells to spherical coordinates, i.e. project each sphere to a (theta, phi) plane \n
    Point 1) gives volumetric 3d data containing a given shell.
    Point 2) gives 2d data in the (theta, phi) plane for a given shell; this can be interpreted as a heatmap.

    Parameters:

    xyz_r: list of list of tuples
        The length of the list corresponds to the number of concentric shells considered.
        Each element in the list - representing a concentric shell - contains a list of 3 dimensional tuples,
        with the indices of the volume elements which belong to the given concentric shell.
        This is the list returned by :py:mod:`ai4materials.utils.utils_vol_data.get_slice_volume_indices`.

    vol_data: numpy.ndarray
        Volumetric data as numpy.ndarray.

    Return: vox_by_slices, theta_phi_by_slices

    vox_by_slices: np.ndarray, shape [n_slices, n_px, n_py, n_pz]
        4-dimensional array containing each concentric shell obtained from
        :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D`.
        ``n_px``, ``n_py``, ``n_pz`` are given by the interpolation and the region of the space
        considered. In our case, ``n_slices=52``, ``n_px=n_py=n_pz=176``.

    theta_phi_by_slices: list of tuples
        Each element in the list correspond to a concentric shell.
        In each concentric shell, there is a list of tuples (theta, phi, intensity) of the non-zero points
        in the volume considered, as return by :py:mod:`ai4materials.utils.utils_vol_data.shells_to_sph`.
        The length of the tuple list of each concentric shell is different because a different number of points
        is non-zero for each shell.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    vox_slice_ri = []
    theta_phi_slice_ri = []

    px, py, pz = vol_data.shape
    x0 = px / 2.0
    y0 = py / 2.0
    z0 = pz / 2.0

    # project each shell onto a 2d surface
    for idx_r, xyz_ri in enumerate(xyz_r):
        vox_slice = np.zeros((px, py, pz))
        cart_coord = np.asarray(xyz_ri).reshape(-1, 3)
        m = np.asarray([x0, y0, z0]).reshape(-1, 3)
        cart_coord = cart_coord - m
        # sph_coord[0] --> r
        # sph_coord[1] --> theta
        # sph_coord[2] --> phi
        sph_coord = _append_spherical_np(cart_coord)[:, 3:]
        theta_phi_slice = []
        for idx_p, item in enumerate(xyz_ri):
            vox_slice[item[0], item[1], item[2]] = vol_data[item[0], item[1], item[2]]
            theta_phi_slice.append((sph_coord[idx_p, 1], sph_coord[idx_p, 2], vol_data[item[0], item[1], item[2]]))
        vox_slice_ri.append(vox_slice)
        theta_phi_slice_ri.append(theta_phi_slice)

    vox_by_slices = np.asarray(vox_slice_ri)
    theta_phi_by_slices = np.asarray(theta_phi_slice_ri)

    return vox_by_slices, theta_phi_by_slices


def interp_theta_phi_surfaces(theta_phi_by_slices_coarse, theta_bins=256, phi_bins=512):
    """Interpolate the spherical shells in spherical coordinate to a finer grid.

    For more information on the interpolation, please refer to:
    http://scipy-cookbook.readthedocs.io/items/Matplotlib_Gridding_irregularly_spaced_data.html

    theta_phi_by_slices_coarse: list of tuples
        Each element in the list correspond to a concentric shell.
        In each concentric shell, there is a list of tuples (theta, phi, intensity) of the non-zero points
        in the volume considered, as return by :py:mod:`ai4materials.utils.utils_vol_data.shells_to_sph`.
        The length of the tuple list of each concentric shell is different because a different number of points
        is non-zero for each shell.

    theta_bins: int, optional (default=256)
        Bins to be used for the interpolation of the theta angle of the spherical coordinates.

    phi_bins: int, optional (default=512)
        Bins to be used for the interpolation of the phi angle of the spherical coordinates.

    Return: np.ndarray, shape [n_slices, theta_bins_fine, phi_bins_fine]
        Three-dimensional array containing each concentric shell in spherical coordinate.
        ``n_slices`` is given by the region of the space considered.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    theta_phi_slice_ri_list_fine = []
    for idx_slice, theta_phi_slice_ri in enumerate(theta_phi_by_slices_coarse.tolist()):
        theta, phi, intensity = zip(*theta_phi_slice_ri)

        # define grid.
        theta_i = np.linspace(0., 180., theta_bins)
        phi_i = np.linspace(-180., 180., phi_bins)

        # grid the data
        intensity_i = griddata((theta, phi), intensity, (theta_i[None, :], phi_i[:, None]), method='cubic')
        intensity_i = np.nan_to_num(intensity_i)

        theta_phi_slice_ri_list_fine.append(intensity_i)

    image_by_slices = np.asarray(theta_phi_slice_ri_list_fine)
    image_by_slices = image_by_slices.reshape(len(theta_phi_by_slices_coarse), theta_bins, phi_bins)

    return image_by_slices


def _slice_3d_volume_slow(vol_data):
    """Obtain slices of 3d volume data. This is very slow and used only for double-checking the fast implementation.

    It is hardcoded to a specific volume size.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    # sample points
    min_r = 5.0
    max_r = 32.0
    dr = 2
    n_bins_r = int((max_r - min_r) / dr) + 1
    theta_bins = 100
    phi_bins = 50

    r = np.linspace(min_r, max_r, n_bins_r)
    phi = np.linspace(0, 2 * np.pi, phi_bins)
    theta = np.linspace(0, np.pi, theta_bins)

    x = 1 * np.outer(np.cos(phi), np.sin(theta))
    y = 1 * np.outer(np.sin(phi), np.sin(theta))
    z = 1 * np.outer(np.ones(np.size(phi)), np.cos(theta))

    px, py, pz = vol_data.shape

    # circle parameters
    x0 = (px - 1.0) / 2.0
    y0 = (py - 1.0) / 2.0
    z0 = (pz - 1.0) / 2.0

    # hardcoded to a specific image size
    max_x = 63
    max_y = 63
    max_z = 63
    assert max_x == max_y

    xyz_r = []
    # the pixels that get hit
    for r_i in r:
        x_sh = r_i * x.flatten()
        y_sh = r_i * y.flatten()
        z_sh = r_i * z.flatten()

        coord = zip(np.rint(x_sh + x0).astype(int), np.rint(y_sh + y0).astype(int), np.rint(z_sh + y0).astype(int))

        xyz = list(set([xyz for xyz in coord if 0 <= xyz[0] < max_x and 0 <= xyz[1] < max_y and 0 <= xyz[2] < max_z]))

        xyz_r.append(xyz)

    vox_slice_ri = []
    theta_phi_slice_ri = []

    for idx, xyz_ri in enumerate(xyz_r):
        vox_slice = np.zeros((px, py, pz))
        theta_phi_slice = []
        for item in xyz_r[idx]:
            vox_slice[item[0], item[1], item[2]] = vol_data[item[0], item[1], item[2]]
            radius = math.sqrt((item[0] - x0) ** 2 + (item[1] - y0) ** 2 + (item[2] - z0) ** 2)
            theta_phi_sel_val = vol_data[item[0], item[1], item[2]]
            theta_sel = math.degrees(math.acos((item[2] - z0) / radius))
            phi_sel = math.degrees(math.atan2(item[1] - y0, item[0] - x0))
            theta_phi_slice.append((theta_sel, phi_sel, theta_phi_sel_val))

        vox_slice_ri.append(vox_slice)
        theta_phi_slice_ri.append(theta_phi_slice)

    vox_by_slices = np.asarray(vox_slice_ri)
    theta_phi_by_slices = np.asarray(theta_phi_slice_ri)

    return vox_by_slices, theta_phi_by_slices
