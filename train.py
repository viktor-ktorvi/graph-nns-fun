import hydra
import torch

import numpy as np

from torch_geometric.transforms import AddSelfLoops, Compose, ToUndirected

from dataset.gaussian_landscape_dataset import GaussianLandscapeDataset
from model.utils import load_model
from train.gnn import train_gnn
from train.mlp import train_mlp
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

    model = load_model(cfg)
    model.to(cfg.training.device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum") if cfg.dataset.task == "classification" else torch.nn.MSELoss(reduction="sum")

    if cfg.model.type == "mlp":
        train_mlp(model, optimizer, criterion, train_dataset, validation_dataset, cfg)
    elif cfg.model.type == "gnn":
        train_gnn(model, optimizer, criterion, train_dataset, validation_dataset, cfg)
    else:
        raise NotImplemented

    debug_var = None


if __name__ == '__main__':
    main()
