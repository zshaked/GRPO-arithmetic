# GRPO for arithmetic

Group Relative Policy Optimization (GRPO) implemented from scratch in PyTorch and run end to end on a CPU: a small transformer learns two-digit addition and subtraction from a 0/1 reward.

## The objective

GRPO comes from [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024, section 4.1). For each prompt `q`, sample a group of `G` completions from the old policy and score each one. Then maximise

```
J = E_q E_{o_i ~ pi_old}  (1/G) sum_i (1/|o_i|) sum_t [
        min( rho_it * A_i,  clip(rho_it, 1 - eps, 1 + eps) * A_i )  -  beta * KL_it
    ]

rho_it = pi_theta(o_it | q, o_i,<t) / pi_old(o_it | q, o_i,<t)
A_i    = (r_i - mean(r)) / std(r)
KL_it  = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
```

The pieces, and where each lives in `grpo.py`:

| Piece | What it does | Function |
| --- | --- | --- |
| Group-relative advantage | Replaces PPO's learned value function with the group's own mean and std. A group whose rewards are all equal has zero advantage. | `group_advantages` |
| Probability ratio and clip | Lets one batch of rollouts be reused for several gradient steps without the policy moving far from the one that sampled them. | `grpo_loss` |
| KL penalty | Keeps the policy near a frozen reference policy. Uses the paper's per-token estimator, which is non-negative. | `kl_estimator` |
| Length normalisation | Averages over a completion's tokens, so long completions do not dominate. | `grpo_loss` |

`tests/test_grpo.py` checks each piece, including that the first gradient step (ratio = 1) equals the plain policy gradient with normalised advantages.

## What changed from the first version

The first version of this repo called its loss GRPO, but it was REINFORCE with a group-mean baseline: `loss = -mean((r_i - mean(r)) * log pi(o_i))`. It had no probability ratio, no clip, no KL penalty and no std normalisation, and it took one gradient step per group. That loss is kept as `reinforce_loss` and used as the baseline below.

## Experiment

- **Task:** all 20,000 problems `a + b` and `a - b` with `a, b` in 0..99, split once into 1,500 warm-start, 16,500 RL and 2,000 test problems.
- **Model:** a 3-layer, 96-dimensional decoder-only transformer over 16 character tokens (`model.py`).
- **Warm start:** 600 supervised steps on the 1,500 solved problems. This leaves the policy right some of the time, which RL from a 0/1 reward needs.
- **RL:** 400 iterations of 64 prompts with 8 completions each. The policy sees only whether each sampled answer was exactly right.
- **Metric:** greedy exact-match accuracy on the 2,000 held-out problems.

| Objective | Seed | Test acc after warm start | Test acc after RL | Gain |
| --- | --- | --- | --- | --- |
| GRPO | 0 | 0.276 | 0.634 | +0.358 |
| GRPO | 1 | 0.113 | 0.340 | +0.227 |
| GRPO | 2 | 0.200 | 0.428 | +0.228 |
| REINFORCE baseline | 0 | 0.276 | 0.543 | +0.267 |
| REINFORCE baseline | 1 | 0.113 | 0.243 | +0.130 |
| REINFORCE baseline | 2 | 0.200 | 0.335 | +0.135 |

Mean gain: GRPO +0.271, REINFORCE baseline +0.177. Per-run histories are in `results/`.

### What this does and does not show

- Both objectives improve held-out accuracy from a 0/1 reward on problems whose answers the policy never sees.
- GRPO gained more on all three seeds, from the same rollouts.
- The comparison is not controlled for compute. GRPO takes 4 gradient steps per batch of rollouts and the baseline takes 1, so the gap may be mostly sample reuse. The clip is what makes that reuse safe, but this experiment does not separate the two.
- Three seeds is too few for a confidence interval, and the warm start varies a lot between seeds (0.11 to 0.28).
- The KL penalty is weak at `beta = 0.04`: the estimator keeps rising through training (about 0.7 by iteration 400). No run tests whether it matters.
- The learning rate and warm-start size were picked from 100-iteration pilot runs on seed 0.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests                      # 9 checks of the objective
python train.py --objective grpo --seed 0   # about 6 minutes on one CPU core
python train.py --objective reinforce --seed 0
python summarize.py                         # table from results/*.json
```

To reproduce the table, run both objectives for seeds 0, 1 and 2 with `--out results/<objective>_seed<seed>.json`. `python train.py --help` lists the hyperparameters.

## Files

| File | Contents |
| --- | --- |
| `grpo.py` | The objective: advantages, clipped surrogate, KL estimator, and the REINFORCE baseline. |
| `model.py` | Character tokenizer and the transformer, with attention written out. |
| `data.py` | Problem generation and the fixed warm/train/test split. |
| `train.py` | Warm start, rollouts, the RL loop and evaluation. |
| `tests/test_grpo.py` | Unit tests for the objective. |
| `results/` | One JSON history per run in the table. |
