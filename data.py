"""Simple arithmetic problem generator for training."""

import random
from typing import List, Dict, Literal

# Type alias for operation types
Operation = Literal['+', '-']


def generate_arithmetic_problems(
    n: int = 100,
    min_value: int = 1,
    max_value: int = 20,
    operations: List[Operation] = ['+', '-']
) -> List[Dict[str, str]]:
    """
    Generate simple arithmetic problems.

    Creates questions like "what is 5 + 3?" with their answers.
    Good for RL training since there's a clear right/wrong signal.
    """
    problems = []

    for _ in range(n):
        operation = random.choice(operations)
        num1 = random.randint(min_value, max_value)
        num2 = random.randint(min_value, max_value)

        question = f"what is {num1} {operation} {num2}?"

        if operation == '+':
            answer = num1 + num2
        elif operation == '-':
            answer = num1 - num2
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        problems.append({
            "question": question,
            "answer": str(answer)
        })

    return problems


if __name__ == "__main__":
    print("Arithmetic Problem Generator\n")

    problems = generate_arithmetic_problems(n=10)

    for i, problem in enumerate(problems, 1):
        question = problem['question']
        answer = problem['answer']
        print(f"{i:2d}. Q: {question:25s} A: {answer:>4s}")

    print(f"\nGenerated {len(problems)} problems")
