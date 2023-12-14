import time

from torch import nn, optim

from datasets.dataset import KarateClubDataset
from models.graph_convolutional_neural_network import GCN
from utils.visualise import visualize_embedding

model_type = GCN
dataset_type = KarateClubDataset


def train_one_epoch(model, data, criterion, optimizer):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss, h


def train_model(
    model: model_type, dataset: dataset_type, num_epochs: int = 400
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    data = dataset.dataset[0]

    for epoch in range(num_epochs):
        loss, h = train_one_epoch(model, data, criterion, optimizer)
        if epoch % 10 == 0:
            visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
            time.sleep(0.3)
