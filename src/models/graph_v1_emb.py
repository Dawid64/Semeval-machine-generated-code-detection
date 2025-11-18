from typing import Literal
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

TASK_CLASS_MAP = {"a": 2, "b": 11, "c": 4}


class GraphV1Emb(nn.Module):
    def __init__(self, task: Literal["a", "b", "c"] = "a"):
        super().__init__()

        num_node_types = 2**16
        embed_dim = 32
        in_channels = 5

        self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)

        in_gnn = embed_dim + in_channels - 1

        self.c1 = GCNConv(in_gnn, 2048)
        self.c2 = GCNConv(2048, 1024)
        self.h1 = nn.Linear(1024, 1024)
        self.h2 = nn.Linear(1024, 512)
        self.o = nn.Linear(512, TASK_CLASS_MAP[task])
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

        x = global_mean_pool(x, batch)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.o(x)
        return x


if __name__ == "__main__":
    from src.train import Trainer

    trainer = Trainer(GraphV1Emb())
    trainer.train()
