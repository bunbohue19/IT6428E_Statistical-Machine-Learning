"""
DPO (Direct Preference Optimization) training on GSM8k.

Offline DPO: preference pairs are constructed statically before training.
  chosen   = the ground-truth chain-of-thought solution (correct)
  rejected = a synthetic version with the final answer perturbed by +1
             (wrong answer but same reasoning format)

Uses TRL's DPOTrainer with a LoRA-adapted Gemma-3-1b-it policy.
The reference policy is implicit (derived from the frozen LoRA-disabled adapters).
"""

import os
import re
import argparse

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

MODEL_ID = "google/gemma-3-1b-it"
DATA_ID  = "openai/gsm8k"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "dpo")

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def make_rejected(answer: str) -> str:
    """Return the answer string with the final numeric answer incremented by 1."""
    m = ANSWER_RE.search(answer)
    if not m:
        return "I don't know the answer.\n#### 0"
    correct = float(m.group(1).replace(",", ""))
    wrong   = int(correct) + 1
    return answer[: m.start(1)] + str(wrong)


def build_dpo_dataset(raw_split, tokenizer, max_samples: int = -1) -> Dataset:
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
        rows.append({
            "prompt":   prompt,
            "chosen":   ex["answer"],
            "rejected": make_rejected(ex["answer"]),
        })

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
    raw = load_dataset(DATA_ID, "main")
    train_ds = build_dpo_dataset(raw["train"], tokenizer, args.num_samples)
    eval_ds  = build_dpo_dataset(raw["test"],  tokenizer, max_samples=200)

    # ── DPO config ───────────────────────────────────────────────────────────
    dpo_cfg = DPOConfig(
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
        remove_unused_columns=False,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # implicit reference via disabled LoRA adapters
        args=dpo_cfg,
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
    parser = argparse.ArgumentParser(description="Offline DPO on GSM8k")
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--beta",        type=float, default=0.1,  help="KL regularization β")
    parser.add_argument("--num_samples", type=int,   default=-1,   help="-1 = full dataset")
    args = parser.parse_args()
    main(args)
