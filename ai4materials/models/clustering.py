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
import inspect
import logging
from ai4materials.dataprocessing.preprocessing import standardize_matrix
from sklearn import cluster
from sklearn import mixture


logger = logging.getLogger('ai4materials')


def design_matrix_to_clustering(design_matrix, clustering_method=None, clustering_class=None,
                                clustering_params=None, standardize='mean-variance'):
    """From a high-dimensional design matrix, obtain clustering values.

    The user can decide to use a pre-defined selection of embedding methods (via ``clustering_method``), or to pass an object
    from the scikit-learn library that can perform the embedding (via ``clustering_class``).

    Parameters:

    design_matrix: ``numpy.ndarray``
        Design matrix. Each row represents an individual object, with the successive columns corresponding to
        the variables and their specific values for that object.

    clustering_method: str, optional, default=`None`, { 'tsne', 'tsomap'}
        Method chosen for the clustering among the pre-selected ones. If `None`, then ``clustering_class`` will be used.

    clustering_class: object
        This is the object that contains the method to calculate the embedding. It must have a `fit_transform`.
        Example of a valid object class is: ``clustering_class`` = `sklearn.manifold.TSNE()`

    clustering_params: dict, optional, default=`None`
        Dictionary containing user-defined parameters to be passed to the objected performing the clustering.
        If the parameters are already present, they are going to be overwritten.

    standardize: str, optional, default='mean-variance', { `None`, 'mean-variance', 'mean', 'variance'}
        Standardize the data or not. See :py:mod:`ai4materials.dataprocessing.preprocessing.standardize_matrix`.

    Return:

    numpy.ndarray, shape [n_samples, 1]
        Returns an array with the clustering labels; one sample for each row, and one column with the clustering
        label.

    object
        Object which contains the actual embedding method. It could be, for example, an object from sklearn.manifold
        (e.g. TSNE, SpectralEmbedding, MDS) or from sklearn.decomposition (e.g. PCA, TruncatedSVD).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
    """

    t0 = time()
    params = {}
    params_gaussian_mixture = {'covariance_type': 'full'}
    params_agglomerative_clustering = {'linkage': 'ward'}
    params_dbscan = {'eps': 0.5, 'min_samples': 50, 'leaf_size': 10}
    params_birch = {'branching_factor': 50, 'threshold': 0.5}

    params_dict = dict(default=params,
                       gaussian_mixture=params_gaussian_mixture,
                       agglomerative_clustering=params_agglomerative_clustering,
                       dbscan=params_dbscan, birch=params_birch)

    clust_dict = dict(kmeans=cluster.KMeans(), gaussian_mixture=mixture.GaussianMixture(),
                      agglomerative_clustering=cluster.AgglomerativeClustering(), dbscan=cluster.DBSCAN(),
                      spectral_clustering=cluster.SpectralClustering(), birch=cluster.Birch())

    if clustering_method is not None:
        if clustering_method in clust_dict.keys():
            if clustering_method in params_dict.keys():
                params = params_dict[clustering_method]
            else:
                params = params_dict['default']

            if clustering_params is not None:
                params.update(clustering_params)

            clustering = clust_dict[clustering_method].set_params(**params)
        else:
            raise Exception("Could not find the chosen embedding method in the pre-defined clustering methods"
                            "Chosen clustering methods: {}. \n"
                            "Pre-defined clustering methods: {}. \n".format(clustering_method, clust_dict.keys()))
    elif clustering_class is not None:
        clustering = clustering_class
    else:
        raise Exception("Please either specified a pre-selected clustering method, or pass an clustering object.")

    design_matrix, scaler = standardize_matrix(design_matrix, standardize=standardize)
    # float32 can fail because of rounding errors
    design_matrix = design_matrix.astype(np.float64)

    logger.info("Clustering method: {}".format(clustering))

    labels, labels_prob, clustering = get_clusters(design_matrix, clustering)

    t1 = time()
    logger.info("Time to compute clustering: {0:.2f} sec".format(t1 - t0))

    return labels, labels_prob, clustering


def get_clusters(data, clustering):
    # get a list of methods of the clustering object
    list_methods = [item[0] for item in inspect.getmembers(clustering, predicate=inspect.ismethod)]

    if 'fit_predict' in list_methods:
        clustering_fit = clustering.fit(data)
        labels = clustering.fit_predict(data)
    elif 'fit' in list_methods and 'predict' in list_methods:
        clustering_fit = clustering.fit(data)
        labels = clustering_fit.predict(data)
    else:
        raise Exception("Could not find 'fit_predict' or ('fit' and 'predict') methods."
                        "Please make sure the method is a valid clustering method.")

    # check if we can compute probabilities
    if 'predict_proba' in list_methods:
        labels_prob = clustering_fit.predict_proba(data)
    else:
        logger.info("The selected clustering method does not allow for the evaluation of clustering probabilities.")
        labels_prob = None

    return labels, labels_prob, clustering
