from model.gcn import GCN
from model.mlp import MLP


def load_model(cfg):
    if cfg.model.name == "mlp":
        return MLP(cfg)
    elif cfg.model.name == "gcn":
        return GCN(cfg)
    else:
        raise NotImplementedError("Model '{:s}' is not supported.".format(cfg.model.name))
