import torch

from torch_geometric.loader import DataLoader
from tqdm import tqdm


def train_gnn(model, optimizer, criterion, train_dataset, validation_dataset, cfg):
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.model.batch_size, shuffle=True)

    progress_bar = tqdm(range(cfg.model.epochs))
    for epoch in progress_bar:

        cumulative_loss_train = 0
        accuracy_count_train = 0
        cumulative_abs_error_train = 0
        cumulative_abs_target_train = 0

        for batch in train_dataloader:
            batch = batch.to(cfg.training.device)

            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index).squeeze()
            loss = criterion(output, batch.y)

            cumulative_loss_train += loss.item()
            prediction = output > 0
            accuracy_count_train += (prediction == batch.y).sum().item()
            cumulative_abs_error_train += torch.abs(output - batch.y).sum().item()
            cumulative_abs_target_train += torch.abs(batch.y).sum().item()

            loss.backward()
            optimizer.step()

        cumulative_loss_val = 0
        accuracy_count_val = 0
        cumulative_abs_error_val = 0
        cumulative_abs_target_val = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                batch = batch.to(cfg.training.device)
                output = model(batch.x, batch.edge_index).squeeze()
                loss = criterion(output, batch.y)

                cumulative_loss_val += loss.item()
                prediction = output > 0
                accuracy_count_val += (prediction == batch.y).sum().item()
                cumulative_abs_error_val += torch.abs(output - batch.y).sum().item()
                cumulative_abs_target_val += torch.abs(batch.y).sum().item()

        if epoch % 1 == 0:
            progress_bar.set_description(
                "GNN training: training loss = {:2.5f}; validation loss = {:2.5f}; train accuracy = {:2.2f}; validation accuracy = {:2.2f}; train relative error = {:2.5f}; validation relative error = {:2.5f}".format(
                    cumulative_loss_train / len(train_dataset),
                    cumulative_loss_val / len(validation_dataset),
                    accuracy_count_train / len(train_dataset) / cfg.topology.nodes,
                    accuracy_count_val / len(validation_dataset) / cfg.topology.nodes,
                    cumulative_abs_error_train / cumulative_abs_target_train,
                    cumulative_abs_error_val / cumulative_abs_target_val
                ))
