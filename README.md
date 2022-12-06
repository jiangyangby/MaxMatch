# MaxMatch

Tensorflow implementation of [MaxMatch](https://arxiv.org/abs/2209.12611) for CIFAR-10, CIFAR-100, SVHN and STL-10 datasets based on the [official FixMatch implementation](https://github.com/google-research/fixmatch). We sincerely thank the authors for their work.

## Setup

**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt
```

### Install datasets

```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
CUDA_VISIBLE_DEVICES= scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
```

## Running

### Setup

All commands must be run from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training a MaxMatch with 32 filters on cifar10 shuffled with `seed=3`, 40 labeled samples and 1 validation sample:
```bash
CUDA_VISIBLE_DEVICES=0 python maxmatch.py --filters=32 --K=3 --dataset=cifar10.3@40-1 --train_dir ./experiments/maxmatch_cta
CUDA_VISIBLE_DEVICES=0 python maxmatch_ra.py --filters=32 --K=3 --dataset=cifar10.3@40-1 --train_dir ./experiments/maxmatch_ra
```

Available labelled sizes are 10, 20, 30, 40, 100, 250, 1000, 4000.
For validation, available sizes are 1, 5000.
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be
shuffled for gradient descent to work properly).

For cifar100, the filters should be `--filters=128`. For stl10, `--scale=4` should be added to adopt WRN-37-2.


#### Multi-GPU training
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python maxmatch.py --filters=32 --dataset=cifar10.3@40-1 --train_dir ./experiments/maxmatch
```

#### Flags

```bash
python maxmatch.py --help
# The following option might be too slow to be really practical.
# python maxmatch.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml maxmatch.py
```

The `--K` flag controls the number of strongly augmented variants. MaxMatch `K=1` is equivalent to FixMatch.

The `--augment` flag can use a little more explanation. It is composed of 3 values, for example `d.d.d`
(`d`=default augmentation, for example shift/mirror, `x`=identity, e.g. no augmentation, `ra`=rand-augment,
 `rac`=rand-augment + cutout):
- the first `d` refers to data augmentation to apply to the labeled example.
- the second `d` refers to data augmentation to apply to the weakly augmented unlabeled example.
- the third `d` refers to data augmentation to apply to the strongly augmented unlabeled example. For the strong
augmentation, `d` is followed by `CTAugment` and code inside `cta/` folder.


### Valid dataset names
```bash
for dataset in cifar10 svhn svhn_noextra; do
for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
for size in 10 20 30 40 100 250 1000 4000; do
    echo "${dataset}.${seed}@${size}-${valid}"
done; done; done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "cifar100.${seed}@10000-${valid}"
done; done

for seed in 1 2 3 4 5; do
for valid in 1 5000; do
    echo "stl10.${seed}@1000-${valid}"
done; done
echo "stl10.1@5000-1"
```


## Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir ./experiments
```

## Checkpoint accuracy

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
./scripts/extract_accuracy.py ./experiments/maxmatch/cifar10.d.d.d.3@40-1/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## Citation

If you use the code of this repository, please cite our paper:
```bibtex
@ARTICLE{jiang22maxmatch,
  author={Jiang, Yangbangyan and Li, Xiaodan and Chen, Yuefeng and He, Yuan and Xu, Qianqian and Yang, Zhiyong and Cao, Xiaochun and Huang, Qingming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={MaxMatch: Semi-Supervised Learning With Worst-Case Consistency},
  year={2022}
}
```
