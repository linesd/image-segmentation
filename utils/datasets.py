import os
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "pascal": "PascalVOC2012"}

DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))

# def get_img_size(dataset):
#     """Return the correct image size."""
#     return get_dataset(dataset).img_size

def get_num_classes(dataset):
    "Return the number of classes"
    return get_dataset(dataset).num_classes

def get_class_labels(dataset):
    """Return the class labels"""
    return get_dataset(dataset).classes

def get_dataloaders(dataset,
                    root=None,
                    shuffle=True,
                    image_set='train',
                    image_size=None,
                    pin_memory=True,
                    batch_size=128,
                    **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset : {"mnist", "fashion"}
        Name of the dataset to load
    root : str
        Path to the dataset root. If `None` uses the default one.
    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)

    if root is None:
        dataset = Dataset(image_set=image_set,
                          image_size=image_size)
    else:
        dataset = Dataset(root=root,
                          image_set=image_set,
                          image_size=image_size)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)

class PascalVOC2012(datasets.VOCSegmentation):
    num_classes = 21
    classes = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor'
    ]

    def __init__(self, root=os.path.join(DIR, '../data/PascalVOC2012'),
                 image_set='train',
                 image_size=(3,100,100)):
        super().__init__(root=root,
                         year='2012',
                         image_set=image_set,
                         download='True',
                         transform=transforms.ToTensor(),
                         target_transform=transforms.ToTensor())

        # transforms.Compose([
        #     transforms.Resize((image_size[1], image_size[2])),
        #     transforms.ToTensor()])