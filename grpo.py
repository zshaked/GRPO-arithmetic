"""
The GRPO objective (Shao et al., 2024, "DeepSeekMath", section 4.1).

For a prompt q, sample a group of G completions o_1..o_G from the old policy
pi_old and score each with a reward r_i. GRPO maximises

    J = E_q E_{o_i ~ pi_old} (1/G) sum_i (1/|o_i|) sum_t [
            min( rho_{i,t} * A_i,  clip(rho_{i,t}, 1-eps, 1+eps) * A_i )
            - beta * KL_{i,t}
        ]

    rho_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_old(o_{i,t} | q, o_{i,<t})
    A_i       = (r_i - mean(r)) / std(r)          (group-relative, no critic)
    KL_{i,t}  = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1

The KL term is the paper's per-token estimator of KL(pi_theta || pi_ref),
evaluated on the sampled token; it is non-negative and zero iff the two
probabilities agree. pi_ref is a frozen reference policy.

Everything here works on per-token log-probabilities, so it does not depend
on the model that produced them.
"""

from dataclasses import dataclass

import torch


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2      # PPO clip range for the probability ratio
    kl_beta: float = 0.04      # weight on the KL penalty to the reference policy
    normalize_std: bool = True  # divide advantages by the group std
    adv_eps: float = 1e-6


def group_advantages(
    rewards: torch.Tensor, normalize_std: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """
    Group-relative advantages.

    rewards: (num_prompts, G). Each row is one prompt's group of completions.
    Returns (num_prompts, G). A group whose rewards are all equal gets all-zero
    advantages: it carries no learning signal.
    """
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize_std:
        return centered
    return centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + eps)


def kl_estimator(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token pi_ref/pi_theta - log(pi_ref/pi_theta) - 1, which is >= 0."""
    log_ratio = logp_ref - logp_new
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Negative GRPO objective, to be minimised.

    logp_new: (N, T) log pi_theta of each sampled token, with gradient
    logp_old: (N, T) log pi_old of the same tokens, recorded at sampling time
    logp_ref: (N, T) log pi_ref of the same tokens
    advantages: (N,) one advantage per completion
    mask: (N, T) 1.0 for real completion tokens (up to and including EOS)
    """
    ratio = torch.exp(logp_new - logp_old.detach())
    adv = advantages.detach().unsqueeze(1)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)

    kl = kl_estimator(logp_new, logp_ref.detach())

    per_token = surrogate - cfg.kl_beta * kl
    lengths = mask.sum(dim=1).clamp(min=1.0)
    per_completion = (per_token * mask).sum(dim=1) / lengths   # (1/|o_i|) sum_t
    loss = -per_completion.mean()

    with torch.no_grad():
        n_tok = mask.sum().clamp(min=1.0)
        outside = ((ratio - 1.0).abs() > cfg.clip_eps).float()
        stats = {
            "kl": ((kl * mask).sum() / n_tok).item(),
            "clip_frac": ((outside * mask).sum() / n_tok).item(),
            "ratio_mean": ((ratio * mask).sum() / n_tok).item(),
        }
    return loss, stats


def reinforce_loss(
    logp_new: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    The baseline this repo used before: REINFORCE with a group-mean baseline.

    loss = -(1/G) sum_i (r_i - mean(r)) * log pi_theta(o_i)

    No probability ratio, no clipping, no KL penalty, no std normalisation,
    and a sum (not a mean) over tokens. Kept for the ablation in train.py.
    rewards: (num_prompts, G); logp_new and mask: (num_prompts * G, T).
    """
    adv = group_advantages(rewards, normalize_std=False).reshape(-1)
    seq_logp = (logp_new * mask).sum(dim=1)
    return -(adv.detach() * seq_logp).mean()
