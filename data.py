"""Two-digit addition and subtraction problems with a fixed train/test split."""

import random

MAX_OPERAND = 99
PROMPT_LEN = 6   # "07+12="
MAX_ANSWER_LEN = 4  # up to "198" or "-99", plus EOS


def make_problem(a: int, b: int, op: str) -> tuple[str, str]:
    """Return (prompt, answer), e.g. ("07+12=", "19"). Operands are zero-padded."""
    answer = a + b if op == "+" else a - b
    return f"{a:02d}{op}{b:02d}=", str(answer)


def all_problems() -> list[tuple[str, str]]:
    return [
        make_problem(a, b, op)
        for op in "+-"
        for a in range(MAX_OPERAND + 1)
        for b in range(MAX_OPERAND + 1)
    ]


def split_problems(
    n_test: int = 2000, n_warm: int = 1500, split_seed: int = 0
) -> tuple[list, list, list]:
    """
    Split all 20,000 problems into (warm, train, test).

    warm:  the only problems the supervised warm start sees, with answers.
    train: problems the RL phase samples prompts from. The policy never sees
           their answers, only a 0/1 reward.
    test:  held out from both phases.
    The split depends only on split_seed, so every run evaluates on the same test set.
    """
    problems = all_problems()
    random.Random(split_seed).shuffle(problems)
    test = problems[:n_test]
    warm = problems[n_test : n_test + n_warm]
    train = problems[n_test + n_warm :]
    return warm, train, test


if __name__ == "__main__":
    warm, train, test = split_problems()
    print(f"warm {len(warm)}  train {len(train)}  test {len(test)}")
    for prompt, answer in test[:5]:
        print(prompt, answer)
