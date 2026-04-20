# SML — LLM Alignment Training Framework

A research implementation comparing five LLM alignment techniques applied to the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) math reasoning benchmark using [Gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) as the base model.

## Methods

| Method | Type | Trainer |
|--------|------|---------|
| **PPO** | On-policy RL (custom) | Custom loop |
| **DPO** | Offline preference | TRL `DPOTrainer` |
| **KTO** | Binary labeling | TRL `KTOTrainer` |
| **RLOO** | On-policy PG + leave-one-out baseline | TRL `RLOOTrainer` |
| **Online-DPO** | Online preference | TRL `OnlineDPOTrainer` |

All methods use LoRA adapters (`r=16, α=32`) on attention projections, keeping ~3–4% of parameters trainable.

## Setup

```bash
pip install -r requirements.txt
cd src
cp .env.example .env   # set HF_HOME if needed
```

Requires a GPU with bfloat16 support. Models and datasets are cached under `huggingface/`.

## Training

```bash
# PPO
python PPO.py [--lr 1e-5] [--batch_size 4] [--epochs 1] [--ppo_epochs 4] \
              [--kl_coef 0.1] [--clip_range 0.2]

# DPO (offline)
python DPO.py [--lr 5e-5] [--batch_size 4] [--grad_accum 4] [--epochs 1] [--beta 0.1]

# KTO
python KTO.py [--lr 5e-5] [--batch_size 4] [--grad_accum 4] [--epochs 1] [--beta 0.1]

# RLOO
python RLOO.py [--lr 1e-5] [--batch_size 4] [--grad_accum 4] [--epochs 1] \
               [--rloo_k 4] [--kl_coef 0.05]

# Online-DPO
python Online-DPO.py [--lr 5e-5] [--batch_size 4] [--grad_accum 4] [--epochs 1] [--beta 0.1]
```

Trained adapters are saved to `results/{method}/`.

## Evaluation

```bash
# Single method
python evaluate.py --model_path ../results/dpo --method DPO \
                   [--num_samples -1] [--batch_size 8] [--max_new_tokens 512]

# All methods
bash evaluate.sh
```

Accuracy is measured by exact numeric match against GSM8k ground-truth answers (extracted via `#### <answer>` pattern).

## Visualization

```bash
python visualize.py [--results_dir ../results] [--out comparison.png]
```

Produces a 2×2 figure (`results/comparison.png`) with training loss, RL mean reward, eval loss, and final accuracy comparison across all methods.

## Project Structure

```
src/
├── PPO.py          # Proximal Policy Optimization
├── DPO.py          # Direct Preference Optimization (offline)
├── KTO.py          # Kahneman-Tversky Optimization
├── RLOO.py         # REINFORCE Leave-One-Out
├── Online-DPO.py   # Online DPO
├── evaluate.py     # Evaluation harness
├── visualize.py    # Result visualization
├── train.sh        # Single-GPU training launcher
└── evaluate.sh     # Batch evaluation script
results/            # Saved adapters and eval_results.json per method
Paper/              # Reference survey paper on LLM alignment
```

## Reference

> A Comprehensive Survey of LLM Alignment Techniques (`Paper/`)
