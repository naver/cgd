# -*- coding: utf-8 -*-
# Copyright 2019-present NAVER Corp.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm

import mxnet as mx
import numpy as np


class Evaluator(object):
    def __init__(self, model, data_loader, ctx):
        self.model = model
        self.data_loader = data_loader
        self.ctx = ctx

    # Reference from Apache MXNet example code
    # (https://github.com/apache/incubator-mxnet/blob/master/example/gluon/embedding_learning/train.py#L123)
    def _evaluate_recall_at_k(self, d_mat, labels, ranks):
        recall_at_ranks = []
        for k in ranks:
            correct, cnt = 0.0, 0.0
            for i in range(d_mat.shape[0]):
                d_mat[i, i] = 1e10
                nns = np.argpartition(d_mat[i], k)[:k]
                if any(labels[i] == labels[nn] for nn in nns):
                    correct += 1
                cnt += 1
            recall_at_ranks.append(correct/cnt)
        return recall_at_ranks

    def evaluate(self, ranks):
        print('Extracting features...')
        test_features, test_class_ids = [], []
        for i, inputs in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            images, class_ids = inputs
            features = self.model(images.as_in_context(self.ctx))
            test_features.append(features.asnumpy())
            test_class_ids.extend(class_ids.asnumpy())
        test_features = np.concatenate(test_features)
        test_class_ids = np.asarray(test_class_ids)

        print('Computing distance matrix...')
        sum_of_squares = np.sum(test_features ** 2.0, axis=1, keepdims=True)
        d_mat = sum_of_squares + sum_of_squares.transpose() - (2.0 * np.dot(test_features, test_features.transpose()))

        print('Evaluating...')
        recall_at_ranks = self._evaluate_recall_at_k(d_mat, test_class_ids, ranks)

        return recall_at_ranks
