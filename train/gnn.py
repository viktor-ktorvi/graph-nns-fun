from torch_geometric.loader import DataLoader


def train_gnn(train_dataset, validation_dataset, cfg):
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # TODO model; optimizer

    # TODO for epoch loop
    #  for train batch loop
    #    zero grad; forward; loss; accumulate metrics;backward; step;
    #  train log
    #  for validation batch loop
    #    forward; loss; accumulate metrics
    #  validation log
    # TODO save model
