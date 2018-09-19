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
__copyright__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "14/08/18"

import unittest
from ai4materials.models.clustering import design_matrix_to_clustering
import numpy as np
import sklearn.manifold
np.random.seed(42)


class TestClustering(unittest.TestCase):
    def setUp(self):
        pass

    def test_design_matrix_to_clustering(self):
        n_samples = 100
        n_dim = 5
        design_matrix = np.random.rand(n_samples, n_dim)

        # test for pre-selected method without user-defined parameters and no probabilities
        labels, labels_prob, clustering = design_matrix_to_clustering(design_matrix, clustering_method='kmeans')
        self.assertIsInstance(labels, np.ndarray)
        self.assertIs(labels_prob, None)

        # test for pre-selected method without user-defined parameters and with probabilities
        # use gaussian_mixture model since it returns also probabilities
        labels, labels_prob, clustering = design_matrix_to_clustering(design_matrix,
                                                                      clustering_method='gaussian_mixture')
        self.assertIsInstance(labels, np.ndarray)
        self.assertGreaterEqual(np.amin(labels_prob), 0.0)
        self.assertLessEqual(np.amax(labels_prob), 1.0)

        # test for pre-selected method without user-defined parameters
        n_clusters = 4
        labels, labels_prob, clustering = design_matrix_to_clustering(design_matrix, clustering_method='kmeans',
                                                                      clustering_params={'n_clusters': n_clusters})
        actual_n_clusters = clustering.get_params()['n_clusters']
        self.assertEqual(actual_n_clusters, n_clusters)
        self.assertIsInstance(labels, np.ndarray)

        # test when a clustering object is directly passed
        dbscan = sklearn.cluster.DBSCAN(eps=0.5, min_samples=50, leaf_size=10)
        clustering_labels, prob_labels, clustering = design_matrix_to_clustering(design_matrix, clustering_class=dbscan)
        self.assertIsInstance(clustering_labels, np.ndarray)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClustering)
    unittest.TextTestRunner(verbosity=2).run(suite)
