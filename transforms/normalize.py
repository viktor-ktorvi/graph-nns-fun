from torch_geometric.transforms import BaseTransform


class Normalize(BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for sample in data:
            sample.x = (sample.x - self.mean) / self.std

        return data
