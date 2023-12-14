import time

import lightning as L
import torch
from torch import nn, optim
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from utils.visualise import visualize_embedding


class GCN(L.LightningModule):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)
        return out, h

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        data = batch
        out, h = self.forward(data.x, data.edge_index)
        out = out.cpu()
        data = data.cpu()
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        if self.current_epoch % 10 == 0:
            visualize_embedding(h, color=data.y, epoch=self.current_epoch, loss=loss)
            time.sleep(0.3)

        return loss
