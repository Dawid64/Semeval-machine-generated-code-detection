from typing import Literal
from torch import nn

from .graph_v1 import GraphClassifier
from .graph_v1_emb import GraphV1Emb
from .graph_v1_emb_large import GraphV1EmbLarge

AVAILABLE_MODELS = Literal["graph_v1", "graph_v1_emb", "graph_v1_emb_large"]

__all_models__: dict[AVAILABLE_MODELS, nn.Module] = {
    "graph_v1": GraphClassifier(),
    "graph_v1_emb": GraphV1Emb(),
    "graph_v1_emb_large": GraphV1EmbLarge(),
}

__all__ = ["GraphClassifier", "GraphV1Emb", "GraphV1EmbLarge", "__all_models__"]
