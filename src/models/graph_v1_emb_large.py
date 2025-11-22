from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphV1EmbLarge(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_node_types: int,
    ):
        super().__init__()

        self.node_type_embedding = nn.Embedding(num_node_types, 128)

        in_gnn = 128 + in_channels - 1

        self.c1 = GCNConv(in_gnn, 2048)
        self.c2 = GCNConv(2048, 1024)
        self.c3 = GCNConv(1024, 1024)
        self.h1 = nn.Linear(1024, 1024)
        self.h2 = nn.Linear(1024, 1024)
        self.h3 = nn.Linear(1024, 512)
        self.o = nn.Linear(512, num_classes)
        self.dropout = 0.1

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, "batch"):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        node_type_ids = x[:, 0].long()
        other_feats = x[:, 1:]

        type_emb = self.node_type_embedding(node_type_ids)

        x = torch.cat([type_emb, other_feats], dim=-1)

        x = self.c1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.c2(x, edge_index)
        x = F.relu(x)

        x = self.c3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = self.o(x)
        return x


if __name__ == "__main__":
    from src.train import Trainer
    from pathlib import Path

    trainer = Trainer(
        GraphV1EmbLarge(5, 2, 65536), save_path=Path("models", "GraphV1EmbLarge.pth")
    )
    trainer.train()
