from __future__ import annotations

import torch
from torch_geometric.data import Data


def graph_to_data(graph) -> Data:
    return Data(
        x=torch.as_tensor(graph.node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(graph.edge_attr, dtype=torch.float32),
    )
