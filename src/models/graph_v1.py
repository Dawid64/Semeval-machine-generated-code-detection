from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GCNConv(5, 2048)
        self.c2 = GCNConv(2048, 1024)
        self.h1 = nn.Linear(1024, 1024)
        self.h2 = nn.Linear(1024, 512)
        self.o = nn.Linear(512, 2)
        self.dropout = 0.1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.c1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.c2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.o(x)
        return x


if __name__ == "__main__":
    from src.train import Trainer

    trainer = Trainer(GraphClassifier())
    trainer.train()
