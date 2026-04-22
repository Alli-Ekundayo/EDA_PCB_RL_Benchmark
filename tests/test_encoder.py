import torch
from torch_geometric.data import Batch, Data

from models.networks import DualStreamActorCritic


def _sample_graph(n_nodes: int = 4):
    x = torch.randn(n_nodes, 6)
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 4)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_dual_stream_forward_shapes():
    model = DualStreamActorCritic(
        node_feat_dim=6,
        edge_feat_dim=4,
        in_channels=7,
        action_dim=32,
        gat_dim=64,
        spatial_dim=64,
        fused_dim=128,
    )
    graphs = Batch.from_data_list([_sample_graph(), _sample_graph(5)])
    spatial = torch.randn(2, 7, 16, 16)
    out = model(graphs, spatial)
    assert out.logits.shape == (2, 32)
    assert out.value.shape == (2,)
    assert out.fused.shape == (2, 128)
    assert not torch.isnan(out.logits).any()
