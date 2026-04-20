"""
RLOO (REINFORCE Leave-One-Out) training on GSM8k.

RLOO is an on-policy policy-gradient algorithm that reduces variance by using
a leave-one-out baseline: for K completions per prompt, the baseline for
completion i is the mean reward of the other K-1 completions.

This implementation uses TRL's RLOOTrainer with:
  - A LoRA-adapted Gemma-3-1b-it as the policy
  - A frozen copy of the base model as the reference policy (KL constraint)
  - A callable rule-based reward function: +1 if the numeric answer matches
    the ground truth (#### N), −1 otherwise
  - K = 4 completions per prompt (args.rloo_k)

The dataset must contain an `input_ids` column (tokenized prompts).
We also store the ground-truth answer in a module-level map so the reward
function can look it up from the decoded full sequence.
"""

import os
import re
import argparse

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

MODEL_ID = "google/gemma-3-1b-it"
DATA_ID  = "openai/gsm8k"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "rloo")

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# Module-level ground-truth map: normalized_question -> answer string
# Populated by build_gt_map() before training starts.
_GT_MAP: dict[str, str] = {}


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def build_gt_map(raw_split) -> None:
    """Populate _GT_MAP with {normalized_question -> answer_string}."""
    global _GT_MAP
    _GT_MAP = {normalize(ex["question"]): ex["answer"] for ex in raw_split}


def rloo_reward_fn(decoded_sequences: list[str]) -> list[float]:
    """
    Callable reward for RLOOTrainer.

    Receives full decoded sequences (prompt + response) and returns a scalar
    reward per sequence.  The question is extracted from the user turn of the
    Gemma chat format so that the ground-truth answer can be looked up.

    Gemma decoded format (skip_special_tokens=True):
        "user\\nQUESTION\\nmodel\\nRESPONSE"
    """
    rewards = []
    for text in decoded_sequences:
        # Strip any remaining special-token markers (<start_of_turn>, etc.)
        clean = re.sub(r"<[^>]+>", "", text)

        # Split into question part and response part on the role boundary
        if "\nmodel\n" in clean:
            q_part, resp = clean.split("\nmodel\n", 1)
        else:
            q_part, resp = clean, ""

        q_part = q_part.replace("user\n", "", 1).strip()
        pred   = extract_answer(resp)

        # Look up GT via prefix match (first 80 normalized chars of question)
        q_norm = normalize(q_part)
        gt_ans = None
        for key, ans in _GT_MAP.items():
            if key[:80] and key[:80] in q_norm:
                gt_ans = extract_answer(ans)
                break

        rewards.append(1.0 if (pred and gt_ans and pred == gt_ans) else -1.0)

    return rewards


def build_rloo_dataset(raw_split, tokenizer, max_samples: int = -1) -> Dataset:
    """
    Build a dataset with `input_ids` column (tokenized prompts).
    RLOOTrainer reads data["input_ids"] directly.
    """
    data = raw_split
    if 0 < max_samples < len(data):
        data = data.select(range(max_samples))

    rows = []
    for ex in data:
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt_str, add_special_tokens=False)
        rows.append({"input_ids": enc["input_ids"]})

    return Dataset.from_list(rows)


def main(args: argparse.Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Policy (trainable) ───────────────────────────────────────────────────
    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    from peft import get_peft_model
    policy = get_peft_model(policy, lora_cfg)
    policy.print_trainable_parameters()

    # ── Reference (frozen) ───────────────────────────────────────────────────
    ref_policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    # ── Datasets ─────────────────────────────────────────────────────────────
    raw = load_dataset(DATA_ID, "main")
    build_gt_map(raw["train"])                         # populate global map
    train_ds = build_rloo_dataset(raw["train"], tokenizer, args.num_samples)
    eval_ds  = build_rloo_dataset(raw["test"],  tokenizer, max_samples=200)

    # ── RLOO config ──────────────────────────────────────────────────────────
    rloo_cfg = RLOOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        rloo_k=args.rloo_k,
        response_length=args.max_new_tokens,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        normalize_reward=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = RLOOTrainer(
        config=rloo_cfg,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=rloo_reward_fn,    # callable: list[str] -> list[float]
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    policy.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLOO on GSM8k")
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--grad_accum",     type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--rloo_k",         type=int,   default=4,   help="Completions per prompt")
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--kl_coef",        type=float, default=0.05)
    parser.add_argument("--num_samples",    type=int,   default=-1)
    args = parser.parse_args()
    main(args)
