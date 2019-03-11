# Combination of Multiple Global Descriptors for Image Retrieval

This is the repository to reproduce the results of our paper **"[Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663)"**.


## Prerequisite

* Python 2.7 or above
* MXNet-1.4.0 or above
* Numpy and tqdm


## Usage

### Download dataset

```console
$ bash download.sh cub200
```


### Extract pre-trained model

```console
$ tar zxvf ./checkpoints/CGD.CUB200.C_concat_MG.ResNet50v.dim1536.tar.gz -C ./checkpoints/
```


### Test

```console
$ python test.py
usage: test.py [-h] [--image-width IMAGE_WIDTH] [--image-height IMAGE_HEIGHT]
               [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
               [--recallk RECALLK] [--data-dir DATA_DIR]
               [--train-txt TRAIN_TXT] [--test-txt TEST_TXT]
               [--bbox-txt BBOX_TXT] --pretrained-model PRETRAINED_MODEL
               [--gpu GPU]
```

```console
$ python test.py --pretrained-model=checkpoints/CGD.CUB200.C_concat_MG.ResNet50v.dim1536
...
R@   1: 0.7681
R@   2: 0.8484
R@   4: 0.9060
R@   8: 0.9433
```


## Citation

```
@article{JunKo3Kim2019,
 Title = {Combination of Multiple Global Descriptors for Image Retrieval},
 Author = {HeeJae Jun and ByungSoo Ko and Youngjoon Kim and Insik Kim and Jongtack Kim},
 Year = {2019},
 Eprint = {arXiv:1903.10663},
}
```


## License

```
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
