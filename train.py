"""
GRPO training for arithmetic tasks.

Trains a language model to solve basic arithmetic using Group Relative Policy Optimization.
GRPO works by generating multiple responses per prompt and comparing them - responses that
do better than average get reinforced, worse ones get penalized.
"""

from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import numpy as np

from data import generate_arithmetic_problems


# Training Configuration Constants
MAX_GENERATION_LENGTH = 20  # Keep responses short for arithmetic
TEMPERATURE = 0.7  # Sampling temperature for generation
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_NUM_SAMPLES = 4  # Number of responses per prompt (the 'k' in GRPO)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Simple container for tracking accuracy during training."""
    epoch: int
    total_correct: int
    total_attempts: int

    @property
    def accuracy(self) -> float:
        return (self.total_correct / self.total_attempts * 100) if self.total_attempts > 0 else 0.0


def extract_answer(response: str, prompt: str) -> str:
    """
    Pull out the number from what the model generated.

    The response includes the original prompt, so we strip that out first.
    Then we look for the first number (including negatives like -5).
    """
    generated_text = response[len(prompt):].strip()
    numbers = re.findall(r"-?\d+", generated_text)
    return numbers[0] if numbers else ""


def compute_rewards(responses: List[str], correct_answer: str) -> np.ndarray:
    """Check which responses are correct. Returns 1.0 for right, 0.0 for wrong."""
    return (np.array(responses) == correct_answer).astype(float)


def compute_advantages(rewards: np.ndarray) -> np.ndarray:
    """
    Mean-center the rewards to get advantages.

    This is how GRPO works - instead of caring about absolute scores, we compare
    responses within each group. Anything above average gets a positive advantage
    (we want more of this), below average gets negative (less of this).
    """
    return rewards - np.mean(rewards)


def generate_with_log_probs(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_length: int = MAX_GENERATION_LENGTH,
    temperature: float = TEMPERATURE
) -> Tuple[str, torch.Tensor]:
    """
    Generate text while keeping track of log probabilities.

    We need the log probs for the policy gradient update. Basically, for each token
    the model generates, we record how confident it was. Later we'll use this to
    make the model more confident in good responses and less confident in bad ones.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = input_ids.clone()
    log_probs = []

    for _ in range(max_length):
        outputs = model(generated_ids)
        next_token_logits = outputs.logits[:, -1, :]
        probs = torch.softmax(next_token_logits / temperature, dim=-1)

        next_token_id = torch.multinomial(probs, num_samples=1)
        log_probs.append(torch.log(probs[0, next_token_id]))

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, requires_grad=True)

    return response, total_log_prob


def train_grpo(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    problems: List[Dict[str, str]],
    num_epochs: int = 2,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    learning_rate: float = DEFAULT_LEARNING_RATE
) -> GPT2LMHeadModel:
    """
    Train using GRPO - Group Relative Policy Optimization.

    For each problem, we generate multiple responses and score them. Then we update
    the model to make better-than-average responses more likely and worse ones less
    likely. It's simpler than PPO because we don't need a value network or clipping,
    just relative comparisons.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Starting GRPO training")
    logger.info(f"  Dataset size: {len(problems)} problems")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Samples per prompt: {num_samples}")
    logger.info(f"  Learning rate: {learning_rate}\n")

    for epoch in range(num_epochs):
        metrics = TrainingMetrics(epoch=epoch + 1, total_correct=0, total_attempts=0)

        for problem_idx, problem in enumerate(problems):
            prompt = f"Q: {problem['question']} A:"
            correct_answer = problem['answer']

            # Sample multiple responses for this problem
            responses = []
            log_probs = []

            for _ in range(num_samples):
                response, log_prob = generate_with_log_probs(
                    model, tokenizer, prompt, max_length=MAX_GENERATION_LENGTH
                )
                answer = extract_answer(response, prompt)
                responses.append(answer)
                log_probs.append(log_prob)

            # Score and compare
            rewards = compute_rewards(responses, correct_answer)
            advantages = compute_advantages(rewards)

            # Policy gradient: push up good responses, push down bad ones
            loss = sum(-adv * lp for adv, lp in zip(advantages, log_probs)) / len(log_probs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics.total_correct += int(rewards.sum())
            metrics.total_attempts += len(rewards)

            del loss, log_probs  # cleanup to avoid memory buildup

            if (problem_idx + 1) % 5 == 0:
                batch_accuracy = rewards.sum() / len(rewards) * 100
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Problem {problem_idx + 1}/{len(problems)} | "
                    f"Batch Accuracy: {batch_accuracy:.1f}%"
                )

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {metrics.epoch} Complete - Accuracy: {metrics.accuracy:.1f}%")
        logger.info(f"{'='*60}\n")

    return model


def demonstrate_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    problems: List[Dict[str, str]],
    num_examples: int = 3
) -> None:
    """Show how the trained model performs on a few problems."""
    logger.info("\n" + "="*60)
    logger.info("MODEL DEMONSTRATION")
    logger.info("="*60 + "\n")

    model.eval()

    with torch.no_grad():
        for i in range(min(num_examples, len(problems))):
            problem = problems[i]
            question = problem['question']
            correct_answer = problem['answer']
            prompt = f"Q: {question} A:"

            logger.info(f"Problem {i + 1}: {question}")
            logger.info(f"Correct Answer: {correct_answer}")
            logger.info("Model Responses:")

            for response_num in range(3):
                response, _ = generate_with_log_probs(
                    model, tokenizer, prompt, max_length=MAX_GENERATION_LENGTH
                )
                answer = extract_answer(response, prompt)
                is_correct = "✓" if answer == correct_answer else "✗"
                logger.info(f"  {response_num + 1}. {answer} {is_correct}")

            logger.info("")

    model.train()


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("GRPO Training: Teaching GPT-2 Basic Arithmetic")
    logger.info("="*60 + "\n")

    logger.info("Loading DistilGPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model loaded\n")

    logger.info("Generating arithmetic problems...")
    problems = generate_arithmetic_problems(n=10)
    logger.info(f"Generated {len(problems)} problems\n")

    logger.info("Sample problems:")
    for i, p in enumerate(problems[:3]):
        logger.info(f"  {i + 1}. {p['question']} = {p['answer']}")
    logger.info("")

    trained_model = train_grpo(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        num_epochs=2,
        num_samples=4,
        learning_rate=5e-5
    )

    demonstrate_model(trained_model, tokenizer, problems)

    output_dir = './grpo_model'
    logger.info(f"Saving model to {output_dir}...")
    trained_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved!\n")

    logger.info("="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
