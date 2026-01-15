# GRPO for Arithmetic

Teaching language models basic arithmetic using Group Relative Policy Optimization.

## What is GRPO?

GRPO is a reinforcement learning algorithm that works by generating multiple responses per prompt and comparing them. Instead of needing a reward model or value network, it just looks at which responses are better than average.

**How it works:**
1. Generate k responses for each prompt
2. Score them (right/wrong for arithmetic)
3. Calculate advantages by comparing to the group mean
4. Update: boost probability of above-average responses, reduce below-average ones

**Why use it:**
- Simpler than PPO (no value network or clipping)
- Works well when there's a clear right/wrong answer
- More stable than vanilla policy gradients
- Efficient - learns from multiple attempts per question

**Core idea:**
```
loss = -1/k * Σ(advantage_i * log_prob(response_i))

where advantage = reward - mean(rewards)
```

## Resources

- [Why GRPO is Important and How It Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/) - Great overview
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948) - Section 3 covers GRPO in detail
- [Hugging Face TRL GRPO Implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
- [OpenAI on RLHF variants](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - Background on policy gradients
- [Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) - Context on why these methods matter

## Project Structure

```
grpo-arithmetic/
├── train.py          # Main GRPO training loop
├── data.py           # Arithmetic problem generator
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install transformers torch numpy

# Run training
python train.py
```

This loads DistilGPT-2, generates 10 problems, trains for 2 epochs, and saves the model.

**What you'll see:**
```
Starting GRPO training
  Dataset size: 10 problems
  Epochs: 2
  Samples per prompt: 4

Epoch 1/2 | Problem 5/10 | Batch Accuracy: 50.0%
Epoch 1/2 | Problem 10/10 | Batch Accuracy: 75.0%

Epoch 1 Complete - Accuracy: 62.5%

...

Problem 1: what is 15 + 8?
Correct Answer: 23
Model Responses:
  1. 23 ✓
  2. 23 ✓
  3. 23 ✓
```

## Configuration

Edit constants in [train.py](train.py):

```python
MAX_GENERATION_LENGTH = 20      # tokens per response
TEMPERATURE = 0.7               # sampling randomness
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_NUM_SAMPLES = 4         # responses per prompt
```

Scale it up:
```python
problems = generate_arithmetic_problems(n=100)
model = train_grpo(model, tokenizer, problems, num_epochs=5, num_samples=8)
```

## How It Works

**1. Generate problems** ([data.py](data.py))
```python
problems = [
    {"question": "what is 5 + 3?", "answer": "8"},
    ...
]
```

**2. Sample responses** ([train.py](train.py))

For each problem, generate k responses while tracking log probs:
```python
for _ in range(num_samples):
    response, log_prob = generate_with_log_probs(model, tokenizer, prompt)
    responses.append(extract_answer(response, prompt))
    log_probs.append(log_prob)
```

**3. Score them**
```python
rewards = compute_rewards(responses, correct_answer)
# [1.0, 1.0, 0.0, 0.0] if 2/4 correct
```

**4. Calculate advantages**
```python
advantages = rewards - np.mean(rewards)
# [0.5, 0.5, -0.5, -0.5]
```

**5. Update policy**
```python
loss = -Σ(advantage * log_prob) / k
loss.backward()
optimizer.step()
```

## Implementation Notes

**Answer extraction:** The model sees `"Q: what is 5 + 3? A:"` and generates `" 8"` or `" the answer is 8"`. We strip the prompt and pull out the first number (regex handles negatives too).

**Log probabilities:** We track the model's confidence in each generated token. This lets us adjust how strongly we push the model toward/away from this response.

**Memory:** Explicitly delete tensors after each update to prevent buildup during training.

## Why This Code

This is written to be readable and educational:

- Type hints throughout
- Clear variable names (`num_samples` not `k`)
- Comments explain the "why" not just the "what"
- Proper logging instead of print statements
- Clean structure that's easy to modify


## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- ~2GB RAM (no GPU needed)

---

Built as a learning project / portfolio piece.
