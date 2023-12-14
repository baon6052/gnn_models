from abc import ABC

import lightning as L
from torch_geometric.datasets import KarateClub
from torch_geometric.loader.dataloader import DataLoader


class AbstractDataset(ABC):
    def __init__(self):
        self.dataset = None

    def print_info(self):
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of features: {self.dataset.num_features}")
        print(f"Number of classes: {self.dataset.num_classes}")

        data = self.dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node "
            f"label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")


class KarateClubDataset(AbstractDataset, L.LightningDataModule):
    def __init__(self):
        self.dataset = KarateClub()
        self.print_info()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)
