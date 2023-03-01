import hydra
import torch

import numpy as np

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddSelfLoops, Compose, ToUndirected

from dataset.gaussian_landscape_dataset import GaussianLandscapeDataset
from transforms.normalize import Normalize
from transforms.utils import get_mean_and_std


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    # set random  seeds
    np.random.seed(cfg.random_seed.numpy)
    torch.random.manual_seed(cfg.random_seed.torch)

    # load and transform the dataset
    dataset = GaussianLandscapeDataset(cfg, transform=Compose([
        AddSelfLoops(),
        ToUndirected()
    ]))

    # split
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    # standard scaling
    normalize = Normalize(*get_mean_and_std(train_dataset))
    train_dataset = normalize(train_dataset)
    validation_dataset = normalize(validation_dataset)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=True)

    # TODO model; optimizer

    # TODO for epoch loop
    #  for train batch loop
    #    zero grad; forward; loss; accumulate metrics;backward; step;
    #  train log
    #  for validation batch loop
    #    forward; loss; accumulate metrics
    #  validation log
    # TODO save model

    debug_var = None


if __name__ == '__main__':
    main()
