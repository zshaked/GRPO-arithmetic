"""Checks that grpo.py computes the objective it claims to."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grpo import GRPOConfig, group_advantages, grpo_loss, kl_estimator, reinforce_loss  # noqa: E402


def test_advantages_are_group_relative():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [5.0, 5.0, 5.0, 9.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(2), atol=1e-3)


def test_uniform_group_has_no_signal():
    adv = group_advantages(torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert torch.equal(adv, torch.zeros(2, 3))


def test_kl_estimator_is_zero_at_reference_and_positive_elsewhere():
    logp = torch.log(torch.tensor([0.5, 0.2]))
    assert torch.allclose(kl_estimator(logp, logp), torch.zeros(2))
    assert (kl_estimator(logp, logp + 0.3) > 0).all()
    assert (kl_estimator(logp, logp - 0.3) > 0).all()


def test_first_step_gradient_is_the_policy_gradient():
    """With pi_theta == pi_old == pi_ref the ratio is 1 and the KL gradient is 0,
    so the gradient must equal that of -(1/N) sum_i A_i * mean_t log pi(o_it)."""
    torch.manual_seed(0)
    logp = torch.randn(6, 4, requires_grad=True)
    adv = torch.randn(6)
    mask = torch.tensor([[1, 1, 1, 0]] * 3 + [[1, 1, 0, 0]] * 3, dtype=torch.float)

    loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, GRPOConfig())
    (grad,) = torch.autograd.grad(loss, logp)

    expected = -(adv.unsqueeze(1) * mask / mask.sum(dim=1, keepdim=True)) / 6
    assert torch.allclose(grad, expected, atol=1e-6)


def test_clip_stops_gradient_once_the_ratio_has_moved_far_enough():
    cfg = GRPOConfig(clip_eps=0.2, kl_beta=0.0)
    mask = torch.ones(2, 1)
    logp_old = torch.zeros(2, 1)
    # Positive advantage, ratio already 1.5 > 1.2: no reward for pushing further.
    # Negative advantage, ratio already 0.5 < 0.8: no reward for pushing further.
    logp_new = torch.log(torch.tensor([[1.5], [0.5]])).requires_grad_(True)
    adv = torch.tensor([1.0, -1.0])
    loss, stats = grpo_loss(logp_new, logp_old, logp_old, adv, mask, cfg)
    (grad,) = torch.autograd.grad(loss, logp_new)
    assert torch.equal(grad, torch.zeros(2, 1))
    assert stats["clip_frac"] == 1.0


def test_clip_does_not_block_moving_back_toward_the_old_policy():
    cfg = GRPOConfig(clip_eps=0.2, kl_beta=0.0)
    # Positive advantage but the ratio fell to 0.5: the unclipped term is the min, gradient flows.
    logp_new = torch.log(torch.tensor([[0.5]])).requires_grad_(True)
    loss, _ = grpo_loss(logp_new, torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([1.0]), torch.ones(1, 1), cfg)
    (grad,) = torch.autograd.grad(loss, logp_new)
    assert grad.item() < 0  # descending the loss raises logp_new


def test_kl_penalty_pulls_toward_the_reference():
    cfg = GRPOConfig(kl_beta=1.0)
    logp_ref = torch.zeros(1, 1)
    logp_new = torch.tensor([[-0.5]], requires_grad=True)  # below the reference
    loss, _ = grpo_loss(logp_new, logp_new.detach(), logp_ref, torch.zeros(1), torch.ones(1, 1), cfg)
    (grad,) = torch.autograd.grad(loss, logp_new)
    assert grad.item() < 0  # descending the loss raises logp_new toward logp_ref


def test_masked_tokens_do_not_contribute():
    logp = torch.randn(2, 3, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    loss, _ = grpo_loss(logp, logp.detach(), logp.detach() + 0.1, torch.tensor([1.0, -1.0]), mask, GRPOConfig())
    (grad,) = torch.autograd.grad(loss, logp)
    assert torch.equal(grad * (1 - mask), torch.zeros(2, 3))


def test_reinforce_baseline_matches_the_old_formula():
    logp = torch.randn(4, 2, requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    mask = torch.ones(4, 2)
    loss = reinforce_loss(logp, rewards, mask)
    seq = logp.sum(dim=1)
    expected = -((rewards[0] - rewards.mean()) * seq).sum() / 4
    assert torch.allclose(loss, expected)
