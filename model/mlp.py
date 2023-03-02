import torch
from torchvision.ops import MLP as MLP_torchvision


class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            MLP_torchvision(
                in_channels=cfg.dataset.features * cfg.topology.nodes,
                hidden_channels=cfg.model.hidden_layers * [cfg.model.hidden_size] + [cfg.topology.nodes]
            ),
            torch.nn.Sigmoid() if cfg.dataset.task == "classification" else lambda x: x
        )

    def forward(self, x):
        return self.mlp(x)
