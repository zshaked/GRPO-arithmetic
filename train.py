"""
GRPO on two-digit arithmetic, end to end on a CPU.

1. Warm start: supervised training of a small transformer on a small set of
   solved problems, so the policy is right some of the time.
2. RL: improve it on unseen problems using only a 0/1 reward, with either the
   GRPO objective (grpo.py) or the REINFORCE baseline this repo used before.
3. Report greedy exact-match accuracy on a held-out test set.

    python train.py --objective grpo --seed 0
    python train.py --objective reinforce --seed 0
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import MAX_ANSWER_LEN, split_problems
from grpo import GRPOConfig, group_advantages, grpo_loss, reinforce_loss
from model import BOS, EOS, PAD, TinyTransformer, decode, encode


def prompt_tensor(prompts: list[str]) -> torch.Tensor:
    return torch.tensor([[BOS] + encode(p) for p in prompts])


def answer_tensor(answers: list[str]) -> torch.Tensor:
    """Answers followed by EOS, right-padded with PAD to MAX_ANSWER_LEN."""
    rows = []
    for a in answers:
        ids = encode(a) + [EOS]
        rows.append(ids + [PAD] * (MAX_ANSWER_LEN - len(ids)))
    return torch.tensor(rows)


def token_logps(model: TinyTransformer, prompts: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """log pi(tokens[:, t] | prompt, tokens[:, :t]) for every t. Shape (N, MAX_ANSWER_LEN)."""
    logits = model(torch.cat([prompts, tokens[:, :-1]], dim=1))
    logits = logits[:, prompts.size(1) - 1 :, :]  # positions that predict the completion
    return torch.log_softmax(logits, dim=-1).gather(2, tokens.unsqueeze(2)).squeeze(2)


@torch.no_grad()
def sample(model: TinyTransformer, prompts: torch.Tensor, greedy: bool = False):
    """
    Sample completions from the policy itself (temperature 1).

    Returns tokens (N, L), their log-probs (N, L), and a mask (N, L) that is 1.0
    up to and including each completion's EOS. Tokens after EOS are PAD.
    """
    n = prompts.size(0)
    ids = prompts
    alive = torch.ones(n, dtype=torch.bool)
    tokens, logps, masks = [], [], []
    for _ in range(MAX_ANSWER_LEN):
        logp_all = torch.log_softmax(model(ids)[:, -1, :], dim=-1)
        if greedy:
            nxt = logp_all.argmax(dim=-1)
        else:
            nxt = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        nxt = torch.where(alive, nxt, torch.full_like(nxt, PAD))
        tokens.append(nxt)
        logps.append(logp_all.gather(1, nxt.unsqueeze(1)).squeeze(1))
        masks.append(alive.float())
        alive = alive & (nxt != EOS)
        ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
    return torch.stack(tokens, 1), torch.stack(logps, 1), torch.stack(masks, 1)


def score(tokens: torch.Tensor, answers: list[str]) -> torch.Tensor:
    """1.0 for an exact-match answer that is terminated by EOS, else 0.0."""
    rewards = []
    for row, answer in zip(tokens.tolist(), answers):
        rewards.append(1.0 if EOS in row and decode(row) == answer else 0.0)
    return torch.tensor(rewards)


def evaluate(model: TinyTransformer, problems: list) -> float:
    model.eval()
    prompts = prompt_tensor([p for p, _ in problems])
    tokens, _, _ = sample(model, prompts, greedy=True)
    model.train()
    return score(tokens, [a for _, a in problems]).mean().item()


def warm_start(model: TinyTransformer, warm: list, steps: int, lr: float, rng: random.Random) -> None:
    """Supervised cross-entropy on the answer tokens of the warm set."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        batch = rng.sample(warm, 128)
        prompts = prompt_tensor([p for p, _ in batch])
        answers = answer_tensor([a for _, a in batch])
        logits = model(torch.cat([prompts, answers[:, :-1]], dim=1))[:, prompts.size(1) - 1 :, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), answers.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    warm, train, test = split_problems(n_warm=args.warm_size)

    model = TinyTransformer()
    warm_start(model, warm, args.warm_steps, args.warm_lr, rng)

    reference = copy.deepcopy(model).eval()  # frozen pi_ref for the KL penalty
    for p in reference.parameters():
        p.requires_grad_(False)

    cfg = GRPOConfig(clip_eps=args.clip_eps, kl_beta=args.kl_beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = [{"iter": 0, "test_acc": evaluate(model, test)}]
    print(f"[{args.objective} seed {args.seed}] after warm start: test acc {history[0]['test_acc']:.3f}")

    start = time.time()
    for it in range(1, args.iters + 1):
        batch = rng.sample(train, args.prompts)
        # Each prompt repeated G times: rows [i*G, (i+1)*G) are prompt i's group.
        prompts = prompt_tensor([p for p, _ in batch]).repeat_interleave(args.group_size, dim=0)
        answers = [a for _, a in batch for _ in range(args.group_size)]

        tokens, logp_old, mask = sample(model, prompts)
        rewards = score(tokens, answers).view(args.prompts, args.group_size)

        if args.objective == "grpo":
            adv = group_advantages(rewards, cfg.normalize_std, cfg.adv_eps).reshape(-1)
            with torch.no_grad():
                logp_ref = token_logps(reference, prompts, tokens)
            # Several gradient steps on the same rollouts. After the first step
            # pi_theta != pi_old, which is what the ratio and the clip are for.
            for _ in range(args.update_epochs):
                loss, stats = grpo_loss(
                    token_logps(model, prompts, tokens), logp_old, logp_ref, adv, mask, cfg
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            loss = reinforce_loss(token_logps(model, prompts, tokens), rewards, mask)
            stats = {}
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            # A group with identical rewards has zero advantage and teaches nothing.
            informative = (rewards.std(dim=1) > 0).float().mean().item()
            row = {
                "iter": it,
                "test_acc": evaluate(model, test),
                "sample_acc": rewards.mean().item(),
                "informative_groups": informative,
                **stats,
            }
            history.append(row)
            extra = "".join(f"  {k} {v:.3f}" for k, v in stats.items())
            print(
                f"[{args.objective} seed {args.seed}] iter {it:4d}  test acc {row['test_acc']:.3f}  "
                f"sample acc {row['sample_acc']:.3f}  informative {informative:.2f}{extra}"
            )

    result = {
        "config": vars(args),
        "history": history,
        "seconds": round(time.time() - start, 1),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", choices=["grpo", "reinforce"], default="grpo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=400, help="RL iterations")
    p.add_argument("--prompts", type=int, default=64, help="prompts per iteration")
    p.add_argument("--group-size", type=int, default=8, help="completions per prompt (G)")
    p.add_argument("--update-epochs", type=int, default=4, help="gradient steps per batch of rollouts (GRPO)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--warm-size", type=int, default=1500, help="solved problems seen by the warm start")
    p.add_argument("--warm-steps", type=int, default=600)
    p.add_argument("--warm-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--out", type=str, default="", help="write the run's history to this JSON file")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
