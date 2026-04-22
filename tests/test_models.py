import torch

from models.actor import MaskedActor
from models.critic import DualHeadCritic


def test_actor_distribution_and_masking():
    actor = MaskedActor(fused_dim=256, board_w=4, board_h=4)
    x = torch.randn(2, 256)
    mask = torch.ones(2, 16)
    mask[:, 0] = 0.0
    dist = actor(x, mask)
    probs = dist.probs
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(probs[:, 0] == 0.0)

    all_invalid = torch.zeros(2, 16)
    dist2 = actor(x, all_invalid)
    probs2 = dist2.probs
    assert torch.allclose(probs2.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_critic_scalar_heads():
    critic = DualHeadCritic(fused_dim=256)
    x = torch.randn(3, 256)
    hpwl, diffp = critic(x)
    assert hpwl.shape == (3, 1)
    assert diffp.shape == (3, 1)
