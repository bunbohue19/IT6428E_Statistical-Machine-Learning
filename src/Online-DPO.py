"""
Online DPO (Direct Preference Optimization) training on GSM8k.

Uses TRL's OnlineDPOTrainer with a rule-based pairwise judge:
  - For each prompt the trainer samples 2 completions from the current policy
  - The judge picks the winner by checking which completion has the correct
    final numeric answer (matching "#### N" in the ground truth)
  - The DPO loss is then computed on the resulting (chosen, rejected) pair

The GSM8k ground-truth mapping is pre-built so the judge can look up the
correct answer from the question text embedded in the decoded prompt.
"""

import os
import re
import argparse

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.judges import BasePairwiseJudge

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

MODEL_ID = "google/gemma-3-1b-it"
DATA_ID  = "openai/gsm8k"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "online_dpo")

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


# ── Rule-based pairwise judge ─────────────────────────────────────────────────

class GSM8kJudge(BasePairwiseJudge):
    """
    Determines which of two completions is better by comparing the extracted
    final numeric answer with the ground truth.

    Scoring:
      +1 (correct) vs −1 (wrong) → correct wins
      Tie (both correct or both wrong) → first completion wins (idx 0)
    """

    def __init__(self, gt_map: dict[str, str]) -> None:
        # gt_map: {normalized_question -> ground_truth_answer_string}
        self._gt_map = gt_map

    def _lookup_gt(self, prompt: str) -> str | None:
        """Find the GT answer for the question contained in *prompt*."""
        norm_prompt = normalize(prompt)
        for q_norm, ans in self._gt_map.items():
            if q_norm in norm_prompt:
                return ans
        return None

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,   # noqa: ARG002 – ignored; rule-based judge is order-agnostic
    ) -> list[int]:
        winners = []
        for prompt, (c0, c1) in zip(prompts, completions):
            gt = self._lookup_gt(prompt)
            if gt is None:
                winners.append(0)
                continue

            gt_ans = extract_answer(gt)
            pred0  = extract_answer(c0)
            pred1  = extract_answer(c1)
            ok0    = pred0 is not None and gt_ans is not None and pred0 == gt_ans
            ok1    = pred1 is not None and gt_ans is not None and pred1 == gt_ans

            if ok0 and not ok1:
                winners.append(0)
            elif ok1 and not ok0:
                winners.append(1)
            else:
                winners.append(0)   # tie → default to first

        return winners


# ── Dataset helpers ──────────────────────────────────────────────────────────

def build_prompt_dataset(raw_split, tokenizer, max_samples: int = -1) -> Dataset:
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
        rows.append({"prompt": prompt})

    return Dataset.from_list(rows)


def build_gt_map(raw_split) -> dict[str, str]:
    """Build {normalized_question -> answer} for the judge."""
    return {normalize(ex["question"]): ex["answer"] for ex in raw_split}


# ── Main ─────────────────────────────────────────────────────────────────────

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

    # ── Datasets & judge ─────────────────────────────────────────────────────
    raw      = load_dataset(DATA_ID, "main")
    gt_map   = build_gt_map(raw["train"])
    train_ds = build_prompt_dataset(raw["train"], tokenizer, args.num_samples)
    eval_ds  = build_prompt_dataset(raw["test"],  tokenizer, max_samples=200)
    judge    = GSM8kJudge(gt_map)

    # ── OnlineDPO config ─────────────────────────────────────────────────────
    cfg = OnlineDPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
        max_length=1024,
        loss_type="sigmoid",        # standard DPO loss
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=None,             # implicit reference via disabled LoRA
        reward_model=None,
        judge=judge,
        args=cfg,
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
    parser = argparse.ArgumentParser(description="Online DPO on GSM8k")
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--batch_size",     type=int,   default=2)
    parser.add_argument("--grad_accum",     type=int,   default=2)
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--beta",           type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=-1)
    args = parser.parse_args()
    main(args)
