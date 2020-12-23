import numpy as np
import os
from PIL import Image
import sys
import torchvision
import pathlib
import requests

from datasets import base
from platforms.platform import get_platform


class CIFAR10(torchvision.datasets.CIFAR10):
    """A subclass to suppress an annoying print statement in the torchvision CIFAR-10 library.

    Not strictly necessary - you can just use `torchvision.datasets.CIFAR10 if the print
    message doesn't bother you.
    """

    def download(self):
        if get_platform().is_primary_process:
            with get_platform().open(os.devnull, 'w') as fp:
                sys.stdout = fp
                super(CIFAR10, self).download()
                sys.stdout = sys.__stdout__
        get_platform().barrier()

# Modified from https://github.com/modestyachts/CIFAR-10.1
def load_new_test_data():
    data_path = get_platform().dataset_root
    filename = os.path.join('cifar10.1', 'v6')
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    # Download the data if it isn't already there
    if not pathlib.Path(label_filepath).is_file():
        url = 'https://raw.github.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy'
        r = requests.get(url)
        os.makedirs(os.path.dirname(label_filepath), exist_ok=True)
        f = open(label_filepath, 'wb')
        f.write(r.content)
    if not pathlib.Path(imagedata_filepath).is_file():
        url = 'https://raw.github.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy'
        r = requests.get(url)
        os.makedirs(os.path.dirname(imagedata_filepath), exist_ok=True)
        f = open(imagedata_filepath, 'wb')
        f.write(r.content)
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    assert labels.shape[0] == 2000
    return imagedata, labels


class Dataset(base.ImageDataset):
    """The CIFAR-10 train set with the CIFAR-10.1 test set."""

    @staticmethod
    def num_train_examples(): return 50000

    @staticmethod
    def num_test_examples(): return 2000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
        train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
        return Dataset(train_set.data, np.array(train_set.targets), augment if use_augmentation else [])

    @staticmethod
    def get_test_set():
        data, labels = load_new_test_data()
        return Dataset(data, labels.astype(np.int64))

    def __init__(self,  examples, labels, image_transforms=None):
        super(Dataset, self).__init__(examples, labels, image_transforms or [],
                                      [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def example_to_image(self, example):
        return Image.fromarray(example)


DataLoader = base.DataLoader
