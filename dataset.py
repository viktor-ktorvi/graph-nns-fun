import hydra
import torch

import networkx as nx
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from typing import Union, List, Tuple

from matplotlib import pyplot as plt

from gaussian_landscape import flip_y_axis, get_node_values, generate_graph, generate_statistics


class GaussianLandscapeDataset(InMemoryDataset):

    def __init__(self, cfg, root="", transform=None):
        super(GaussianLandscapeDataset, self).__init__(root, transform)

        self.graph, self.pos_dict = generate_graph(cfg.topology.nodes, cfg.topology.radius, cfg.topology.threshold, cfg.random_seed.networkx)

        x = []
        for i in range(cfg.features):
            feature_means, feature_covs = generate_statistics(cfg.gaussians)
            x.append(get_node_values(self.graph, self.pos_dict, feature_means, feature_covs))

        target_means, target_covs = generate_statistics(cfg.gaussians)
        y = torch.tensor(get_node_values(self.graph, self.pos_dict, target_means, target_covs))

        self.data_basis = Data(
            x=torch.tensor(x).T,
            y=y,
            edge_index=torch.tensor(list(self.graph.edges)).T,
            pos=torch.tensor([self.pos_dict[node] for node in self.graph]))

    def download(self):
        pass

    def draw(self, **kwargs):
        nx.draw(self.graph, flip_y_axis(self.pos_dict), **kwargs)

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
