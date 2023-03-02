import torch


def to_tensor_dataset(graph_dataset):
    in_tensor = torch.vstack([sample.x.flatten()[None, :] for sample in graph_dataset])
    target_tensor = torch.vstack([sample.y[None, :] for sample in graph_dataset])

    return torch.utils.data.TensorDataset(in_tensor, target_tensor)
