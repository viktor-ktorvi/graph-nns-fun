import copy
import hydra

import networkx as nx
import numpy as np

from matplotlib import cm
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt


def generate_statistics(cfg):
    """
    Generate random means and cov matrices.
    :param cfg: config
    :return: multiple means and cov matrices
    """
    means = np.empty(shape=(0, cfg.dimension))
    covs = np.empty(shape=(0, cfg.dimension, cfg.dimension))
    for i in range(cfg.number):
        means = np.vstack((means, np.random.uniform(low=cfg.mean.low, high=cfg.mean.high, size=(1, cfg.dimension))))
        covs = np.vstack((covs, np.eye(cfg.dimension)[np.newaxis, :] * np.random.uniform(low=cfg.cov.low, high=cfg.cov.high)))

    return means, covs


def generate_landscape(X, Y, cfg):
    """
    Generate a 'mountainous' landscape by adding a bunch of gaussians together.
    :param X: meshgrid
    :param Y: meshgrid
    :param cfg: config
    :return: landscape, means and covs used to generate the landscape
    """
    Z = np.zeros_like(X)

    means, covs = generate_statistics(cfg)

    for i in range(means.shape[0]):
        Z += multivariate_normal.pdf(np.dstack((X, Y)), means[i], covs[i])

    return Z, means, covs


def get_landscape_value(x, y, means, covs):
    """
    Read a single value from the gaussian landscape.
    :param x: float
    :param y: float
    :param means: means of the gaussian landscape
    :param covs: cov matrices of the gaussian landscape
    :return: float
    """
    z = 0
    for i in range(means.shape[0]):
        z += multivariate_normal.pdf(np.dstack((x, y)), means[i], covs[i])

    return z


def translate_positions(graph):
    """
    Networkx gives the positions in a [0, 1] range. We translate it to the [-1, 1] range.
    :param graph: networkx graph
    :return: node positions dict
    """
    pos = nx.get_node_attributes(graph, "pos")  # [0, 1]
    for node in pos:
        for i in range(len(pos[node])):
            pos[node][i] -= 0.5
            pos[node][i] *= 2

    return pos  # [-1, 1]


def flip_y_axis(pos):
    """
    Networkx draws graphs with a flipped y-axis. We flip it here as a counter measure.
    :param pos: node positions dict
    :return: node positions dict with a flipped y-axis
    """
    new_pos = copy.deepcopy(pos)
    for node in new_pos:
        new_pos[node][1] *= -1

    return new_pos


def get_node_values(graph, pos, means, covs):
    """
    For each node in a graph we read its positions value in the gaussian landscape.
    :param graph: networkx graph
    :param pos: positions dict
    :param means: means of the gaussian landscape
    :param covs: cov matrices of the gaussian landscape
    :return: array of the values in node order
    """
    node_values = []
    for node in graph:
        node_values.append(get_landscape_value(pos[node][0], pos[node][1], means, covs))

    return np.array(node_values)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    np.random.seed(cfg.random_seed.numpy)

    resolution = cfg.plotting.resolution
    xlim = cfg.plotting.xlim
    ylim = cfg.plotting.ylim

    # random geometric graph
    graph = nx.random_geometric_graph(cfg.topology.nodes, cfg.topology.radius, seed=cfg.random_seed.networkx)
    pos = translate_positions(graph)  # positions from [0, 1] to [-1, 1]
    pos_for_plotting = flip_y_axis(pos)  # nx.draw flips the y-axis, so we have to flip it back

    # meshgrid for all future plots
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], resolution),
                       np.linspace(ylim[0], ylim[1], resolution))

    # 3D landscape for visualization
    Z, means, covs = generate_landscape(X, Y, cfg.gaussians)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title("Gaussian landscape - 3D")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                    linewidth=0, antialiased=False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plot features, targets and graphs
    fig, ax = plt.subplots(2, cfg.features + 2, sharey="row")
    fig.suptitle("Features and targets")

    ax[0, 0].set_ylabel("y")  # align y axis
    ax[0, 0].set_ylim(ylim)

    # features
    for i in range(cfg.features):
        # feature maps
        Z, means, covs = generate_landscape(X, Y, cfg.gaussians)
        ax[0, i].imshow(Z, cmap=cm.viridis, aspect="auto", extent=xlim + ylim, alpha=1)
        ax[0, i].set_title("# {:d}".format(i))
        ax[0, i].set_xlabel("x")

        # graph with feature map values
        nx.draw(graph, pos_for_plotting, node_color=get_node_values(graph, pos, means, covs), cmap=cm.viridis, vmin=np.min(Z), vmax=np.max(Z), ax=ax[1, i])

    # regression target
    Z_regression_target, means, covs = generate_landscape(X, Y, cfg.gaussians)

    ax[0, cfg.features].imshow(Z_regression_target, cmap=cm.coolwarm, aspect="auto", extent=xlim + ylim, alpha=1)
    ax[0, cfg.features].set_title("regression\ntarget")
    ax[0, cfg.features].set_xlabel("x")
    nx.draw(graph, pos_for_plotting, node_color=get_node_values(graph, pos, means, covs), cmap=cm.coolwarm, vmin=np.min(Z_regression_target), vmax=np.max(Z_regression_target),
            ax=ax[1, cfg.features])

    # classification target
    Z_classification_target, means, covs = generate_landscape(X, Y, cfg.gaussians)

    # TODO possibly do multiple class scenario

    mean_Z = np.mean(Z_classification_target)  # subtract the mean and threshold
    Z_classification_target -= mean_Z
    Z_classification_target[Z_classification_target <= 0] = 0
    Z_classification_target[Z_classification_target > 0] = 1

    ax[0, cfg.features + 1].imshow(Z_classification_target, aspect="auto", cmap=cm.coolwarm, extent=xlim + ylim, alpha=1)
    ax[0, cfg.features + 1].set_title("classification\ntarget")

    node_values = get_node_values(graph, pos, means, covs)
    node_values -= mean_Z
    node_values[node_values <= 0] = 0
    node_values[node_values > 0] = 1

    nx.draw(graph, pos_for_plotting, node_color=node_values, cmap=cm.coolwarm, vmin=np.min(Z_classification_target), vmax=np.max(Z_classification_target),
            ax=ax[1, cfg.features + 1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # TODO generate a dataset by multiplying the node vars with [0.9, 1.1] or similar
