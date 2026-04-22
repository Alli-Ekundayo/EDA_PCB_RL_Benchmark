from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except Exception as exc:  # pragma: no cover
    Batch = None
    GATv2Conv = None
    global_mean_pool = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class GATEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, embed_dim: int = 128, heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        if GATv2Conv is None:  # pragma: no cover
            raise ImportError("torch-geometric is required for GATEncoder") from IMPORT_ERROR

        self.conv1 = GATv2Conv(node_feat_dim, 64, heads=heads, edge_dim=edge_feat_dim, dropout=0.1)
        self.conv2 = GATv2Conv(64 * heads, embed_dim, heads=1, edge_dim=edge_feat_dim, dropout=0.1)
        self.norm1 = nn.LayerNorm(64 * heads)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, batch_graph: Batch) -> torch.Tensor:
        x = batch_graph.x
        edge_index = batch_graph.edge_index
        edge_attr = batch_graph.edge_attr
        b = batch_graph.batch

        h = F.elu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))
        return global_mean_pool(h, b)
