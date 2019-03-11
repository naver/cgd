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

import mxnet as mx
import os


class ImageData(mx.gluon.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        image, image_id, bbox = self.dataset[item]
        image = mx.image.imread(image)

        if bbox is not None:
            x, y, w, h = bbox
            image = image[y:min(y+h, image.shape[0]), x:min(x+w, image.shape[1])]

        if self.transform is not None:
            image = self.transform(image)

        return image, image_id

    def __len__(self):
        return len(self.dataset)


class Dataset(object):
    def __init__(self, dataset_dir, train_txt, test_txt, bbox_txt=None):
        self.dataset_dir = dataset_dir
        self._bbox = None

        if bbox_txt is not None:
            self._load_bbox(bbox_txt)

        train, num_train_ids = self._load_meta(train_txt)
        test, num_test_ids = self._load_meta(test_txt)

        self.train = train
        self.test = test

        self.num_train_ids = num_train_ids
        self.num_test_ids = num_test_ids

    def _load_bbox(self, bbox_file):
        self._bbox = {}
        with open(bbox_file, 'r') as f:
            for line in f:
                image_id, x, y, w, h = line.strip().split()
                self._bbox[int(image_id)] = [ int(float(x)), int(float(y)), int(float(w)), int(float(h)) ]

    def _load_meta(self, meta_file):
        datasets = []
        prev_label = -1
        num_class_ids = 0
        with open(meta_file, 'r') as f:
            for line_no, line in enumerate(f):
                if line_no == 0:
                    continue

                image_id, label, image_path = line.strip().split()
                if self._bbox is not None:
                    bbox = self._bbox[int(image_id)]
                else:
                    bbox = None
                datasets.append((os.path.join(self.dataset_dir, image_path), int(label), bbox))
                if prev_label != int(label):
                    num_class_ids += 1
                    prev_label = int(label)

        return datasets, num_class_ids

    def print_stats(self):
        num_total_ids = self.num_train_ids + self.num_test_ids

        num_train_images = len(self.train)
        num_test_images = len(self.test)
        num_total_images = num_train_images + num_test_images

        print("###### Dataset Statistics ######")
        print("+------------------------------+")
        print("| Subset  | #Classes | #Images |")
        print("+------------------------------+")
        print("| Train   |    {:5d} | {:7d} |".format(self.num_train_ids, num_train_images))
        print("| Test    |    {:5d} | {:7d} |".format(self.num_test_ids, num_test_images))
        print("+------------------------------+")
        print("| Total   |    {:5d} | {:7d} |".format(num_total_ids, num_total_images))
        print("+------------------------------+")
