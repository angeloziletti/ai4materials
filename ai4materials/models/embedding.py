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

from time import time
import numpy as np
from scipy.sparse import csr_matrix
import sklearn.manifold
from sklearn import decomposition
from ai4materials.dataprocessing.preprocessing import standardize_matrix
import logging

logger = logging.getLogger('ai4materials')


def design_matrix_to_embedding(design_matrix, embed_method=None, embed_class=None,
                               embed_params=None, standardize='mean-variance', n_components=2):
    """From a high-dimensional design matrix, reduce the dimensionality via embedding methods.

    The user can decide to use a pre-defined selection of embedding methods (via ``embed_method``), or to pass an object
    from the scikit-learn library that can perform the embedding (via ``embed_class``).

    Parameters:

    design_matrix: ``numpy.ndarray``
        Design matrix. Each row represents an individual object, with the successive columns corresponding to
        the variables and their specific values for that object.

    embed_method: str, optional, default=`None`, { 'tsne', 'tsne_pca', 'tsne_approx', 'spect_embed', 'mds', 'isomap', 'hessian', 'pca', 'ipca', 'kpca', 'svd'}
        Method chosen for the embedding among the pre-selected ones. If `None`, then ``embed_class`` will be used.

    embed_class: object
        This is the object that contains the method to calculate the embedding. It must have a `fit_transform`.
        Example of a valid object class is: ``embed_class`` = `sklearn.manifold.TSNE()`

    embed_params: dict, optional, default=`None`
        Dictionary containing user-defined parameters to be passed to the objected performing the embedding.
        If the parameters are already present, they are going to be overwritten.

    standardize: str, optional, default='mean-variance', { `None`, 'mean-variance', 'mean', 'variance'}
        Standardize the data or not. See :py:mod:`ai4materials.dataprocessing.preprocessing.standardize_matrix`.

    Return:

    numpy.ndarray, shape [n_samples, 2]
        Returns an array with the embedding coordinates; one sample for each row, and two columns corresponding to the
        coordinates.

    object
        Object which contains the actual embedding method. It could be, for example, an object from sklearn.manifold
        (e.g. TSNE, SpectralEmbedding, MDS) or from sklearn.decomposition (e.g. PCA, TruncatedSVD).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    t0 = time()
    params = {'n_components': n_components}
    params_tsne = {'method': 'exact', 'init': 'random', 'random_state': 42, 'verbose': 0, 'n_iter': 1000}
    params_tsne_pca = {'method': 'exact', 'init': 'pca', 'random_state': 42, 'verbose': 0}
    params_tsne_approx = {'method': 'barnes_hut', 'init': 'pac', 'random_state': 42, 'verbose': 0}

    params_dict = dict(default=params,
                       tsne=params_tsne, tsne_pca=params_tsne_pca, tsne_approx=params_tsne_approx)

    embed_dict = dict(tsne=sklearn.manifold.TSNE(),
                      tsne_pca=sklearn.manifold.TSNE(),
                      tsne_approx=sklearn.manifold.TSNE(),
                      spect_embed=sklearn.manifold.SpectralEmbedding(),
                      mds=sklearn.manifold.MDS(),
                      isomap=sklearn.manifold.Isomap(),
                      hessian=sklearn.manifold.LocallyLinearEmbedding(),
                      pca=decomposition.PCA(),
                      ipca=decomposition.IncrementalPCA(),
                      kpca=decomposition.KernelPCA(),
                      svd=decomposition.TruncatedSVD())

    if embed_method is not None:
        if embed_method in embed_dict.keys():
            if embed_method in params_dict.keys():
                params = params_dict[embed_method]
            else:
                params = params_dict['default']

            if embed_params is not None:
                params.update(embed_params)

            embedding = embed_dict[embed_method].set_params(**params)
        else:
            raise Exception("Could not find the chosen embedding method in the pre-defined embedding methods"
                            "Chosen embedding methods: {}. \n"
                            "Pre-defined embedding methods: {}. \n".format(embed_method, embed_dict.keys()))
    elif embed_class is not None:
        embedding = embed_class
    else:
        raise Exception("Please either specified a pre-selected embedding method, or pass an embedding object.")

    design_matrix, scaler = standardize_matrix(design_matrix, standardize=standardize)
    # float32 can fail because of rounding errors
    design_matrix = design_matrix.astype(np.float64)

    logger.info("Embedding method: {}".format(embedding))
    mapping = embedding.fit_transform(design_matrix)

    # special output
    if embed_method == 'pca':
        tot_exp_var = sum(embedding.explained_variance_ratio_)
        logger.info('Explained variance by each component (%):{}'.format(embedding.explained_variance_ratio_ * 100.0))
        logger.info('Total variance explained (%): {}'.format(tot_exp_var * 100.0))
        logger.info('Eigenvectors: ')
        logger.info('Eigenvector 1: \n{}'.format(embedding.components_[0]))
        logger.info('Eigenvector 2: \n{}'.format(embedding.components_[1]))

    t1 = time()
    logger.info("Time to compute 2d-embedding: {0:.2f} sec".format(t1 - t0))

    return mapping, embedding


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
