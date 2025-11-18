from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_node_types: int,
        embed_dim: int = 32,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.node_type_embedding = nn.Embedding(num_node_types, embed_dim)

        in_gnn = embed_dim + in_channels - 1

        self.c1 = GCNConv(in_gnn, 2048)
        self.c2 = GCNConv(2048, 1024)
        self.h1 = nn.Linear(1024, 1024)
        self.h2 = nn.Linear(1024, 512)
        num_outputs = self.num_classes if self.num_classes > 2 else 1
        self.o = nn.Linear(512, num_outputs)
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
        if self.num_classes > 2:
            x = nn.Softmax()(x)
        return x


if __name__ == "__main__":
    from src.train import Trainer

    trainer = Trainer(GraphClassifier(5, 2, 65536))
    trainer.train()
