"""
Evaluation script for GSM8k alignment-trained models.

Loads a fine-tuned model (base model + LoRA adapters saved by any of the
training scripts), generates chain-of-thought solutions for every question
in the GSM8k test split, and reports exact-match accuracy on the final
numeric answer (the number following "#### ").

Usage
-----
    python evaluate.py --model_path ../results/dpo --method DPO
    python evaluate.py --model_path ../results/ppo --method PPO
    python evaluate.py --model_path google/gemma-3-1b-it --method baseline

Output
------
    Prints per-batch progress and a final accuracy/summary to stdout.
    Also saves a JSON results file to <model_path>/eval_results.json.
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_ID            = "openai/gsm8k"

NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"

GT_ANSWER_RE = re.compile(rf"####\s*({NUM})")

# Ordered fallback patterns for model predictions. First match wins.
PRED_PATTERNS = [
    re.compile(rf"####\s*({NUM})"),                          # GSM8k style
    re.compile(rf"\\boxed\{{\s*({NUM})\s*\}}"),              # \boxed{N}
    re.compile(rf"(?:final\s+)?answer\s*(?:is|:)\s*\**\s*({NUM})", re.IGNORECASE),
    re.compile(rf"=\s*\**\s*({NUM})\s*\**\s*\.?\s*$", re.MULTILINE),
]

ANY_NUM_RE = re.compile(NUM)


def _clean(s: str) -> str:
    return s.replace(",", "").lstrip("$").rstrip(".")


def _numeric_eq(a: str, b: str) -> bool:
    """Compare two numeric strings with tolerance for trailing zeros / decimals."""
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def extract_gt_answer(text: str) -> str | None:
    """Strict: only accept the GSM8k ``####`` format (used for ground truth)."""
    m = GT_ANSWER_RE.search(text)
    return _clean(m.group(1)) if m else None


def extract_pred_answer(text: str) -> str | None:
    """Tolerant: try common answer formats, fall back to the last number."""
    for pat in PRED_PATTERNS:
        m = None
        for m in pat.finditer(text):
            pass   # keep the last match (the final answer, not an intermediate)
        if m:
            return _clean(m.group(1))
    matches = ANY_NUM_RE.findall(text)
    return _clean(matches[-1]) if matches else None


def load_model_and_tokenizer(
    model_path: str,
    base_model_id: str | None,
    device: torch.device,
):
    """
    Load tokenizer and model.

    If *model_path* points to a directory with a PEFT adapter_config.json,
    load the base model first and then attach the LoRA adapters. The base
    model is taken from ``adapter_config.json['base_model_name_or_path']``
    unless *base_model_id* is explicitly provided (CLI override).

    Otherwise treat *model_path* as a standalone HuggingFace model ID / path.
    """
    adapter_config = os.path.join(model_path, "adapter_config.json")

    if os.path.isfile(adapter_config):
        if base_model_id is None:
            with open(adapter_config) as f:
                base_model_id = json.load(f)["base_model_name_or_path"]
        print(f"Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16
        )
        print(f"Attaching LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()   # merge for faster inference
    else:
        print(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    model.to(device)
    return model, tokenizer


def generate_answers(
    model,
    tokenizer,
    questions: list[str],
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.0,    # greedy by default
    batch_size: int = 8,
) -> list[str]:
    """Generate one answer per question and return the decoded response strings."""
    all_responses: list[str] = []

    SYSTEM_PROMPT = (
        "Solve the math problem step by step. "
        "At the end, write your final answer on its own line in the format: #### <number>"
    )

    for start in range(0, len(questions), batch_size):
        batch_qs = questions[start : start + batch_size]
        prompts  = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_qs
        ]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs["do_sample"] = False   # greedy

            gen_ids = model.generate(**enc, **gen_kwargs)

        response_ids = gen_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        all_responses.extend(decoded)

    return all_responses


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.base_model, device
    )

    # ── Load test set ─────────────────────────────────────────────────────
    raw      = load_dataset(DATA_ID, "main")
    test_ds  = raw["test"]
    if 0 < args.num_samples < len(test_ds):
        test_ds = test_ds.select(range(args.num_samples))

    questions = test_ds["question"]
    gt_answers = [extract_gt_answer(a) for a in test_ds["answer"]]

    print(f"\nEvaluating {len(questions)} test examples …\n")
    t0 = time.time()

    # ── Generate ─────────────────────────────────────────────────────────
    responses = generate_answers(
        model, tokenizer, questions, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # ── Score ─────────────────────────────────────────────────────────────
    correct, total = 0, 0
    failures: list[dict] = []

    for i, (resp, gt) in enumerate(zip(responses, gt_answers)):
        pred = extract_pred_answer(resp)
        ok   = pred is not None and gt is not None and _numeric_eq(pred, gt)
        if ok:
            correct += 1
        else:
            failures.append({
                "idx":       i,
                "question":  questions[i],
                "gt_answer": gt,
                "predicted": pred,
                "response":  resp[-800:],   # tail is where the final answer lives
            })
        total += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{total}] running accuracy = {correct/(i+1)*100:.1f}%")

    elapsed = time.time() - t0
    accuracy = correct / total if total else 0.0

    print(f"\n{'='*50}")
    print(f"Method       : {args.method}")
    print(f"Model path   : {args.model_path}")
    print(f"Test samples : {total}")
    print(f"Correct      : {correct}")
    print(f"Accuracy     : {accuracy*100:.2f}%")
    print(f"Elapsed      : {elapsed:.1f}s  ({elapsed/total:.2f}s/sample)")
    print(f"{'='*50}\n")

    results = {
        "method":      args.method,
        "model_path":  args.model_path,
        "num_samples": total,
        "correct":     correct,
        "accuracy":    accuracy,
        "elapsed_s":   elapsed,
        "failures":    failures[:20],   # keep first 20 for inspection
    }

    # ── Save results ──────────────────────────────────────────────────────
    out_file = os.path.join(args.model_path, "eval_results.json")
    try:
        os.makedirs(args.model_path, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_file}")
    except OSError as e:
        print(f"Warning: could not save results file – {e}")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8k test set")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to fine-tuned model dir (with adapter_config.json) OR HF model ID",
    )
    parser.add_argument(
        "--method", type=str, default="unknown",
        help="Label for this run (e.g. PPO, DPO, KTO, RLOO, Online-DPO, baseline)",
    )
    parser.add_argument("--base_model",     type=str,   default=None,
                        help="Override base model ID. Default: read from adapter_config.json.")
    parser.add_argument("--num_samples",    type=int,   default=-1,   help="-1 = full test set")
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--max_new_tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.0,  help="0 = greedy decoding")
    args = parser.parse_args()
    evaluate(args)
