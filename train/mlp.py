import torch

from tqdm import tqdm

from train.utils import to_tensor_dataset


def train_mlp(model, optimizer, criterion, train_dataset, validation_dataset, cfg):
    train_dataset = to_tensor_dataset(train_dataset)
    validation_dataset = to_tensor_dataset(validation_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=cfg.model.batch_size, shuffle=True)

    progress_bar = tqdm(range(cfg.model.epochs))

    for epoch in progress_bar:
        cumulative_loss_train = 0
        accuracy_count_train = 0

        for batch in train_dataloader:
            x = batch[0].to(cfg.training.device)
            y = batch[1].to(cfg.training.device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.float())

            cumulative_loss_train += loss.item()
            prediction = output > 0
            accuracy_count_train += (prediction == y).sum().item()

            loss.backward()
            optimizer.step()

        cumulative_loss_val = 0
        accuracy_count_val = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                x = batch[0].to(cfg.training.device)
                y = batch[1].to(cfg.training.device)

                output = model(x)
                loss = criterion(output, y.float())
                cumulative_loss_val += loss.item()
                prediction = output > 0
                accuracy_count_val += (prediction == y).sum().item()

        if epoch % 1 == 0:
            progress_bar.set_description("MLP training: training loss = {:2.5f}; validation loss = {:2.5f}; train accuracy = {:2.2f}; validation accuracy = {:2.2f}".format(
                cumulative_loss_train / len(train_dataset),
                cumulative_loss_val / len(validation_dataset),
                accuracy_count_train / len(train_dataset) / cfg.topology.nodes,
                accuracy_count_val / len(validation_dataset) / cfg.topology.nodes,
            ))

        # TODO log
