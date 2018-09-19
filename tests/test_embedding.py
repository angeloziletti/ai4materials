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
from ai4materials.models.embedding import design_matrix_to_embedding
from ai4materials.models.embedding import standardize_matrix
import numpy as np
import sklearn.manifold
from sklearn import preprocessing
np.random.seed(42)


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        pass

    def test_design_matrix_to_embedding(self):
        n_samples = 100
        n_dim = 5
        design_matrix = np.random.rand(n_samples, n_dim)

        # test for pre-selected method without user-defined parameters
        mapping, embedding = design_matrix_to_embedding(design_matrix, embed_method='tsne')
        self.assertIsInstance(mapping, np.ndarray)

        # test for pre-selected method with user-defined parameters
        learning_rate = 100
        mapping, embedding = design_matrix_to_embedding(design_matrix, embed_method='tsne',
                                                        embed_params=dict(learning_rate=learning_rate))
        actual_learning_rate = embedding.get_params()['learning_rate']
        self.assertEqual(actual_learning_rate, learning_rate)
        self.assertIsInstance(mapping, np.ndarray)

        # test for pre-selected method without user-defined parameters
        learning_rate = 100
        mapping, embedding = design_matrix_to_embedding(design_matrix, embed_method='tsne',
                                                        embed_params=None)
        actual_learning_rate = embedding.get_params()['learning_rate']
        self.assertNotEqual(actual_learning_rate, learning_rate)
        self.assertIsInstance(mapping, np.ndarray)

        # test when an embedding object is directly passed
        tsne = sklearn.manifold.TSNE()
        mapping, embedding = design_matrix_to_embedding(design_matrix, embed_class=tsne)
        self.assertIsInstance(mapping, np.ndarray)

    def test_standardize_matrix(self):
        matrix = np.random.rand(3, 2)
        std_matrix, scaler = standardize_matrix(matrix, standardize='mean-variance')
        self.assertAlmostEqual(np.mean(std_matrix), 0.0)
        self.assertAlmostEqual(np.std(std_matrix), 1.0)
        self.assertIsInstance(scaler, preprocessing.StandardScaler)

        matrix = np.random.rand(100, 20)
        std_matrix, scaler = standardize_matrix(matrix, standardize='mean')
        self.assertAlmostEqual(np.mean(std_matrix), 0.0)
        self.assertIsInstance(scaler, preprocessing.StandardScaler)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbedding)
    unittest.TextTestRunner(verbosity=2).run(suite)
