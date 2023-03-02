import torch

from sklearn.preprocessing import StandardScaler


def get_mean_and_std(dataset):
    x = torch.vstack([data.x for data in dataset])

    scaler = StandardScaler()
    scaler.fit(x.cpu().numpy())

    return torch.tensor(scaler.mean_, dtype=torch.float32), torch.tensor(scaler.scale_, dtype=torch.float32)
