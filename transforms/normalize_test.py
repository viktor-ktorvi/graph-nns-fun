import torch
import unittest

import numpy as np

from hydra import initialize, compose

from dataset.gaussian_landscape_dataset import GaussianLandscapeDataset
from transforms.normalize import Normalize
from transforms.utils import get_mean_and_std


def datasets_and_transform(cfg):
    np.random.seed(cfg.random_seed.numpy)
    torch.random.manual_seed(cfg.random_seed.torch)

    dataset = GaussianLandscapeDataset(cfg)

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    transform = Normalize(*get_mean_and_std(train_dataset))

    return train_dataset, validation_dataset, transform


class TestNormalization(unittest.TestCase):

    def test_train_val_split_normalization(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="default")
            train_dataset, validation_dataset, transform = datasets_and_transform(cfg)

        places = 2

        def assertAlmostEqual(torch_array, target_val):
            for val in torch_array:
                self.assertAlmostEqual(val.item(), target_val, places=places)

        x_train = torch.vstack([data.x for data in train_dataset])
        print("\nTrain before")
        print("Mean: ", x_train.mean(dim=0), "\nStd: ", x_train.std(dim=0))

        train_dataset = transform(train_dataset)

        x_train_ = torch.vstack([data.x for data in train_dataset])
        print("\nTrain after")
        print("Mean: ", x_train_.mean(dim=0), "\nStd: ", x_train_.std(dim=0))

        assertAlmostEqual(x_train_.mean(dim=0), 0)
        assertAlmostEqual(x_train_.std(dim=0), 1)

        x_validation = torch.vstack([data.x for data in validation_dataset])
        print("\nValidation before")
        print("Mean: ", x_validation.mean(dim=0), "\nStd: ", x_validation.std(dim=0))

        validation_dataset = transform(validation_dataset)

        x_validation_ = torch.vstack([data.x for data in validation_dataset])
        print("\nValidation after")
        print("Mean: ", x_validation_.mean(dim=0), "\nStd: ", x_validation_.std(dim=0))

        assertAlmostEqual(x_validation_.mean(dim=0), 0)
        assertAlmostEqual(x_validation_.std(dim=0), 1)


if __name__ == '__main__':
    unittest.main()
