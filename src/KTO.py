"""
KTO (Kahneman-Tversky Optimization) training on GSM8k.

KTO replaces pairwise preference data with binary (desirable/undesirable) labels:
  desirable   (label=True)  → ground-truth chain-of-thought solutions
  undesirable (label=False) → synthetic responses with a wrong final answer

The dataset is intentionally balanced: equal counts of positive and negative
examples as recommended in the KTO paper (Ethayarajh et al., 2024).

Uses TRL's KTOTrainer with a LoRA-adapted Gemma-3-1b-it.
"""

import os
import re
import argparse
import random

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_ID  = "openai/gsm8k"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "kto")

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def make_wrong_answer(answer: str, rng: random.Random) -> str:
    """Create a wrong completion by replacing the final numeric answer."""
    m = ANSWER_RE.search(answer)
    if not m:
        return "The answer is unknown.\n#### 0"
    correct = float(m.group(1).replace(",", ""))
    # Perturb by a small offset (avoid returning the correct answer)
    offsets = [-2, -1, 1, 2, 3, -3]
    offset  = rng.choice(offsets)
    wrong   = int(correct) + offset
    return answer[: m.start(1)] + str(wrong)


def build_kto_dataset(
    raw_split,
    tokenizer,
    max_samples: int = -1,
    seed: int = 42,
) -> Dataset:
    rng  = random.Random(seed)
    data = raw_split
    if 0 < max_samples < len(data):
        data = data.select(range(max_samples))

    rows = []
    for ex in data:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Positive example: ground-truth answer
        rows.append({
            "prompt":     prompt,
            "completion": ex["answer"],
            "label":      True,
        })

        # Negative example: wrong answer (same format, wrong final number)
        rows.append({
            "prompt":     prompt,
            "completion": make_wrong_answer(ex["answer"], rng),
            "label":      False,
        })

    # Shuffle so positives and negatives are interleaved
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def main(args: argparse.Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    raw      = load_dataset(DATA_ID, "main")
    train_ds = build_kto_dataset(raw["train"], tokenizer, args.num_samples)
    eval_ds  = build_kto_dataset(raw["test"],  tokenizer, max_samples=200)

    # ── KTO config ───────────────────────────────────────────────────────────
    kto_cfg = KTOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        beta=args.beta,
        max_prompt_length=512,
        max_length=1024,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = KTOTrainer(
        model=model,
        ref_model=None,             # implicit reference via disabled LoRA
        args=kto_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KTO on GSM8k")
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--beta",        type=float, default=0.1,
                        help="KL regularization β (desirable) – undesirable uses β/2")
    parser.add_argument("--num_samples", type=int,   default=-1,
                        help="Max training samples (-1 = full dataset); doubles for neg examples")
    args = parser.parse_args()
    main(args)
