import copy
import hydra
import torch

import networkx as nx
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from typing import Union, List, Tuple

from matplotlib import pyplot as plt

from gaussian_landscape import flip_y_axis, get_node_values, generate_graph, generate_statistics


class GaussianLandscapeDataset(InMemoryDataset):

    def __init__(self, cfg, root="", transform=None):
        super(GaussianLandscapeDataset, self).__init__(root, transform)

        self.data_samples = []
        self.graph, self.pos_dict = generate_graph(cfg.topology.nodes, cfg.topology.radius, cfg.topology.threshold, cfg.random_seed.networkx)

        x = []
        for i in range(cfg.dataset.features):
            feature_means, feature_covs = generate_statistics(cfg.gaussians)
            x.append(get_node_values(self.graph, self.pos_dict, feature_means, feature_covs))

        target_means, target_covs = generate_statistics(cfg.gaussians)
        y = torch.tensor(get_node_values(self.graph, self.pos_dict, target_means, target_covs))

        self.data_basis = Data(
            x=torch.tensor(np.array(x)).T,
            y=y,
            edge_index=torch.tensor(list(self.graph.edges)).T,
            pos=torch.tensor([self.pos_dict[node] for node in self.graph]))

        self.generate_data(cfg.dataset.size)

        if cfg.dataset.task == "classification":
            self.classification_transform()

        elif cfg.dataset.task == "regression":
            pass

        else:
            raise ValueError("Dataset task(='{:s}') is not supported. The supported tasks are: regression and classification".format(cfg.dataset.task))

    def classification_transform(self):
        y_mean = self.data_basis.y.mean()
        for i in tqdm(range(len(self.data_samples)), ascii=True, desc="Creating classification targets"):
            class_mask = self.data_samples[i].y <= y_mean
            self.data_samples[i].y[class_mask] = 0
            self.data_samples[i].y[~class_mask] = 1

    def download(self):
        pass

    def draw(self, **kwargs):
        nx.draw(self.graph, flip_y_axis(self.pos_dict), **kwargs)

    def get(self, idx: int) -> Data:
        assert idx < self.len(), "Index(={:d}) is higher then the length of the dataset(={:d})".format(idx, self.len())

        return self.data_samples[idx]

    def generate_data(self, num, low=0.9, high=1.1):
        """
        Generate the samples by multiplying the features and targets with noise.
        :param num: int; size of the dataset
        :param low: float; low value of the uniform distribution
        :param high: float; high value of the uniform distribution
        :return:
        """
        for i in tqdm(range(num), ascii=True, desc="Generating data"):
            sample = copy.deepcopy(self.data_basis)
            sample.x *= torch.FloatTensor(*sample.x.shape).uniform_(low, high)
            sample.y *= torch.FloatTensor(*sample.y.shape).uniform_(low, high)

            self.data_samples.append(sample)

    def len(self) -> int:
        return len(self.data_samples)

    def process(self):
        pass

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ""

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ""


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    dataset = GaussianLandscapeDataset(cfg)

    plt.figure()
    dataset.draw()
    plt.show()


if __name__ == "__main__":
    main()
