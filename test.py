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

import argparse
import mxnet as mx

from mxnet.gluon.data.vision import transforms

from dataset import ImageData, Dataset

from evaluator import Evaluator


class Model(object):
    def __init__(self, opt):
        sym, arg_params, aux_params = mx.model.load_checkpoint(opt.pretrained_model, 0)
        self._data_shape = (opt.batch_size, 3, opt.image_height, opt.image_height)
        self._executor = sym.simple_bind(ctx=opt.ctx, data=self._data_shape, grad_req='null', force_rebind=True)
        self._executor.copy_params_from(arg_params, aux_params)

    def __call__(self, data):
        if self._data_shape != data.shape:
            new_shape = { 'data': data.shape }
            self._data_shape = data.shape
            self._executor = self._executor.reshape(partial_shaping=True, allow_up_sizing=True, **new_shape)
        y = self._executor.forward(is_train=False, data=data.as_in_context(opt.ctx))
        embeds = y[0]
        return embeds


def test(opt):
    # Load dataset
    dataset = Dataset(opt.data_dir, opt.train_txt, opt.test_txt, opt.bbox_txt)
    dataset.print_stats()

    # Load image transform
    test_transform = transforms.Compose([
        transforms.Resize((opt.image_width, opt.image_height)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load data loader
    test_loader = mx.gluon.data.DataLoader(
        dataset=ImageData(dataset.test, test_transform),
        batch_size=opt.batch_size,
        num_workers=opt.num_workers
    )

    # Load model
    model = Model(opt)

    # Load evaluator
    evaluator = Evaluator(model, test_loader, opt.ctx)

    # Evaluate
    recalls = evaluator.evaluate(ranks=opt.recallk)
    for recallk, recall in zip(opt.recallk, recalls):
        print("R@{:4d}: {:.4f}".format(recallk, recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CGD.')
    parser.add_argument('--image-width',        type=int,       default=224)
    parser.add_argument('--image-height',       type=int,       default=224)
    parser.add_argument('--batch-size',         type=int,       default=1)
    parser.add_argument('--num-workers',        type=int,       default=1)
    parser.add_argument('--recallk',            type=str,       default='1,2,4,8')
    parser.add_argument('--data-dir',           type=str,       default='./data/CUB_200_2011')
    parser.add_argument('--train-txt',          type=str,       default='./meta/CUB200/train.txt')
    parser.add_argument('--test-txt',           type=str,       default='./meta/CUB200/test.txt')
    parser.add_argument('--bbox-txt',           type=str,       default='./meta/CUB200/bbox.txt')
    parser.add_argument('--pretrained-model',   type=str,       required=True)
    parser.add_argument('--gpu',                type=int,       default=0)
    
    opt = parser.parse_args()

    opt.recallk = [ int(k) for k in opt.recallk.split(',') ]
    opt.ctx = mx.cpu() if opt.gpu < 0 else mx.gpu(opt.gpu)

    test(opt)
