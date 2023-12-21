import lightning as L
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GATConv

from utils.visualise import visualize_embedding


class GAT(L.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_hidden: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.6,
        visualise: bool = False,
    ):
        super().__init__()

        self.conv1 = GATConv(
            num_features, num_hidden, heads=num_heads, dropout=dropout
        )
        self.conv2 = GATConv(
            num_hidden * num_heads,
            num_classes,
            concat=False,
            heads=1,
            dropout=dropout,
        )

        self.visualise = visualise
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.005)
        return optimizer

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        if self.visualise and self.current_epoch % 10 == 0:
            visualize_embedding(
                h, color=batch.y, epoch=self.current_epoch, loss=loss
            )

        return loss
