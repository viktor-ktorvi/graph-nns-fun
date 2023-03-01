import copy
import hydra
import torch

import networkx as nx
import numpy as np

from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from typing import Union, List, Tuple

from matplotlib import cm
from matplotlib import pyplot as plt

from gaussian_landscape import feature_target_plot, flip_y_axis, get_landscape_value, get_node_values, generate_graph, generate_statistics


class GaussianLandscapeDataset(InMemoryDataset):
    # TODO how does the saving to disc with/without transforms work?
    def __init__(self, cfg, root="", transform=None):
        super(GaussianLandscapeDataset, self).__init__(root, transform)

        self.data = []
        self.graph, self.pos_dict = generate_graph(cfg.topology.nodes, cfg.topology.radius, cfg.topology.threshold, cfg.random_seed.networkx)

        self.feature_means = []
        self.feature_covs = []
        x = []
        for i in range(cfg.dataset.features):
            feature_means, feature_covs = generate_statistics(cfg.gaussians)
            self.feature_means.append(feature_means)
            self.feature_covs.append(feature_covs)

            x.append(get_node_values(self.graph, self.pos_dict, feature_means, feature_covs))

        self.target_means, self.target_covs = generate_statistics(cfg.gaussians)
        y = torch.tensor(get_node_values(self.graph, self.pos_dict, self.target_means, self.target_covs))

        self.data_basis = Data(
            x=torch.tensor(np.array(x)).T,
            y=y,
            edge_index=torch.tensor(list(self.graph.edges)).T,
            pos=torch.tensor([self.pos_dict[node] for node in self.graph]))

        self.generate_data(cfg.dataset.size)

        self.task = cfg.dataset.task
        if cfg.dataset.task == "classification":
            self.classification_transform()

        elif cfg.dataset.task == "regression":
            pass

        else:
            raise ValueError("Dataset task(='{:s}') is not supported. The supported tasks are: regression and classification".format(cfg.dataset.task))

    def classification_transform(self):
        y_mean = self.data_basis.y.mean()
        for i in tqdm(range(len(self.data)), ascii=True, desc="Creating classification targets"):
            class_mask = self.data[i].y <= y_mean
            self.data[i].y[class_mask] = 0
            self.data[i].y[~class_mask] = 1

    def download(self):
        pass

    def draw_basis(self, **kwargs):
        feature_target_plot(self.graph, self.pos_dict,
                            self.feature_means, self.feature_covs,
                            self.target_means, self.target_covs,
                            **kwargs)

    def draw_graph(self, **kwargs):
        plt.title("Graph")
        nx.draw(self.graph, flip_y_axis(self.pos_dict), **kwargs)

    def draw_sample(self, idx, xlim=(-1, 1), ylim=(-1, 1), resolution=100, cmap_features=cm.viridis, cmap_targets=cm.coolwarm):
        sample = self.get(idx)
        num_features = sample.x.shape[1]

        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], resolution),
                           np.linspace(ylim[0], ylim[1], resolution))

        fig, ax = plt.subplots(1, num_features + 1, sharey="row")
        fig.suptitle("Features and target of sample #{:d}".format(idx))

        for i in range(num_features):
            ax[i].set_title("# {:d}".format(i))

            # feature maps
            feature_landscape = get_landscape_value(X, Y, self.feature_means[i], self.feature_covs[i])

            # graph with feature map values
            nx.draw(
                self.graph, flip_y_axis(self.pos_dict),
                node_color=sample.x[:, i].cpu().numpy(),
                cmap=cmap_features,
                vmin=np.min(feature_landscape),
                vmax=np.max(feature_landscape),
                ax=ax[i]
            )

        # target
        target = sample.y.cpu().numpy()

        if self.task == "regression":
            regression_landscape = get_landscape_value(X, Y, self.target_means, self.target_covs)
            vmin = np.min(regression_landscape)
            vmax = np.max(regression_landscape)
        elif self.task == "classification":
            vmin = np.min(target)
            vmax = np.max(target)
        else:
            raise NotImplemented

        ax[-1].set_title("target")
        nx.draw(
            self.graph, flip_y_axis(self.pos_dict),
            node_color=target,
            cmap=cmap_targets,
            vmin=vmin,
            vmax=vmax,
            ax=ax[-1]
        )

    def get(self, idx: int) -> Data:
        assert idx < self.len(), "Index(={:d}) is higher then the length of the dataset(={:d})".format(idx, self.len())

        return self.data[idx]

    def get_split(self, ratios):
        """
        Split dataset.
        :param ratios: list of ratios of subsets
        :return: tuple of subsets
        """
        return torch.utils.data.random_split(self, ratios)

    def generate_data(self, num, low=0.9, high=1.1):
        """
        Generate the samples by multiplying the features and targets with noise.
        :param num: int; size of the dataset
        :param low: float; low value of the uniform distribution
        :param high: float; high value of the uniform distribution
        :return:
        """
        for _ in tqdm(range(num), ascii=True, desc="Generating data"):
            sample = copy.deepcopy(self.data_basis)
            sample.x *= torch.FloatTensor(*sample.x.shape).uniform_(low, high)
            sample.y *= torch.FloatTensor(*sample.y.shape).uniform_(low, high)

            self.data.append(sample)

    def len(self) -> int:
        return len(self.data)

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
    np.random.seed(cfg.random_seed.numpy)
    torch.random.manual_seed(cfg.random_seed.torch)

    dataset = GaussianLandscapeDataset(cfg)

    plt.figure()
    dataset.draw_graph()

    dataset.draw_basis(
        cmap_features=cm.viridis, cmap_targets=cm.coolwarm,
        resolution=cfg.plotting.resolution,
        xlim=cfg.plotting.xlim, ylim=cfg.plotting.ylim
    )

    dataset.draw_sample(
        7,
        cmap_features=cm.viridis, cmap_targets=cm.coolwarm,
        resolution=cfg.plotting.resolution,
        xlim=cfg.plotting.xlim, ylim=cfg.plotting.ylim
    )

    dataset.draw_sample(
        779,
        cmap_features=cm.viridis, cmap_targets=cm.coolwarm,
        resolution=cfg.plotting.resolution,
        xlim=cfg.plotting.xlim, ylim=cfg.plotting.ylim
    )

    plt.show()


if __name__ == "__main__":
    main()
