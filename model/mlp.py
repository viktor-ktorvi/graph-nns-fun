import torch
import torch_geometric


class MLP(torch.nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.mlp = torch_geometric.nn.MLP(
            in_channels=cfg.dataset.features * cfg.topology.nodes,
            hidden_channels=cfg.model.hidden_size,
            out_channels=cfg.topology.nodes,
            num_layers=cfg.model.hidden_layers,
            dropout=cfg.model.dropout
        )

    def forward(self, x):
        return self.mlp(x)
