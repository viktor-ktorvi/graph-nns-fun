import torch
import torch_geometric


class GCN(torch.nn.Module):
    def __init__(self, cfg):
        super(GCN, self).__init__()

        self.conv = torch_geometric.nn.GCN(
            in_channels=cfg.dataset.features,
            hidden_channels=cfg.model.hidden_size,
            num_layers=cfg.model.hidden_layers, out_channels=1
        )

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
