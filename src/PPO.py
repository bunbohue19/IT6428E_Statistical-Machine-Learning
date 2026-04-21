"""
PPO (Proximal Policy Optimization) training on GSM8k.

Uses AutoModelForCausalLMWithValueHead with a custom PPO loop:
  - Policy + value head share the language model backbone
  - Reference policy is a frozen copy (for KL penalty)
  - Reward: rule-based exact-match on the final numeric answer (#### N)
  - Advantage: clipped scalar reward minus KL penalty (no GAE for simplicity)
"""

import os
import re
import json
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, create_reference_model

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "huggingface"))

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_ID   = "openai/gsm8k"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "ppo")

ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def compute_rewards(responses: list[str], ground_truths: list[str]) -> torch.Tensor:
    rewards = []
    for resp, gt in zip(responses, ground_truths):
        pred   = extract_answer(resp)
        gt_ans = extract_answer(gt)
        rewards.append(1.0 if (pred and gt_ans and pred == gt_ans) else -1.0)
    return torch.tensor(rewards, dtype=torch.float32)


def build_prompts(questions: list[str], tokenizer) -> list[str]:
    prompts = []
    for q in questions:
        p = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(p)
    return prompts


def sequence_logprobs(
    logits: torch.Tensor,       # (B, T, V)
    input_ids: torch.Tensor,    # (B, T)
    response_mask: torch.Tensor # (B, T)
) -> torch.Tensor:
    """Sum of log-probabilities over response tokens."""
    lp = F.log_softmax(logits[:, :-1, :], dim=-1)          # (B, T-1, V)
    token_lp = lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (token_lp * response_mask[:, 1:]).sum(-1)        # (B,)


def main(args: argparse.Namespace) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── LoRA config ──────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Policy (trainable) ───────────────────────────────────────────────────
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        peft_config=lora_cfg,
    )

    # ── Reference (frozen) ───────────────────────────────────────────────────
    ref_policy = create_reference_model(policy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    ref_policy.to(device)
    ref_policy.eval()

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = load_dataset(DATA_ID, "main")
    train_ds = ds["train"]
    if 0 < args.num_samples < len(train_ds):
        train_ds = train_ds.select(range(args.num_samples))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    log_history: list[dict] = []

    for epoch in range(args.epochs):
        policy.train()
        total_reward, total_loss, n_steps = 0.0, 0.0, 0

        for start in range(0, len(train_ds), args.batch_size):
            batch        = train_ds[start : start + args.batch_size]
            questions    = batch["question"]
            ground_truths = batch["answer"]

            prompts = build_prompts(questions, tokenizer)
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            # ── Generate responses ────────────────────────────────────────────
            with torch.no_grad():
                gen_ids = policy.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )

            full_ids    = gen_ids                          # (B, prompt+resp)
            attn_mask   = (full_ids != tokenizer.pad_token_id).long()
            resp_mask   = torch.zeros_like(attn_mask)
            resp_mask[:, prompt_len:] = 1
            resp_mask   = resp_mask & attn_mask

            responses = tokenizer.batch_decode(
                full_ids[:, prompt_len:], skip_special_tokens=True
            )
            rewards = compute_rewards(responses, ground_truths).to(device)

            # ── Old log-probs (fixed for ratio computation) ────────────────
            with torch.no_grad():
                old_logits, _, _ = policy(
                    input_ids=full_ids, attention_mask=attn_mask
                )
                old_lp = sequence_logprobs(old_logits, full_ids, resp_mask)

                ref_logits, _, _ = ref_policy(
                    input_ids=full_ids, attention_mask=attn_mask
                )
                ref_lp = sequence_logprobs(ref_logits, full_ids, resp_mask)

            kl = (old_lp - ref_lp).detach()               # approximate KL
            advantages = rewards - args.kl_coef * kl
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ── PPO mini-epochs ────────────────────────────────────────────
            for _ in range(args.ppo_epochs):
                new_logits, _, new_values = policy(
                    input_ids=full_ids, attention_mask=attn_mask
                )
                new_lp = sequence_logprobs(new_logits, full_ids, resp_mask)

                ratio         = torch.exp(new_lp - old_lp.detach())
                clipped_ratio = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                policy_loss   = -torch.min(
                    ratio * advantages.detach(),
                    clipped_ratio * advantages.detach(),
                ).mean()

                # Value loss: predict scalar reward for each response token
                val_seq    = new_values[:, :-1].squeeze(-1)  # (B, T-1)
                val_target = rewards.unsqueeze(1).expand_as(val_seq) * resp_mask[:, 1:].float()
                value_loss = F.mse_loss(
                    val_seq * resp_mask[:, 1:].float(), val_target
                )

                loss = policy_loss + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            total_reward += rewards.mean().item()
            total_loss   += loss.item()
            n_steps      += 1

            log_history.append({
                "step":        n_steps,
                "epoch":       epoch + start / len(train_ds),
                "loss":        loss.item(),
                "mean_reward": rewards.mean().item(),
            })

            if n_steps % 20 == 0:
                print(
                    f"[epoch {epoch+1}] step {n_steps:4d} | "
                    f"reward={total_reward/n_steps:.3f} | loss={total_loss/n_steps:.4f}"
                )

        print(
            f"Epoch {epoch+1}/{args.epochs} — "
            f"avg_reward={total_reward/max(n_steps,1):.4f}"
        )

    policy.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    with open(os.path.join(OUT_DIR, "train_log.json"), "w") as f:
        json.dump({"log_history": log_history}, f, indent=2)

    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO on GSM8k")
    parser.add_argument("--lr",             type=float, default=1e-5)
    parser.add_argument("--batch_size",     type=int,   default=2)
    parser.add_argument("--epochs",         type=int,   default=1)
    parser.add_argument("--ppo_epochs",     type=int,   default=4,   help="PPO mini-epochs per rollout batch")
    parser.add_argument("--max_new_tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--kl_coef",        type=float, default=0.1,  help="KL penalty coefficient")
    parser.add_argument("--clip_range",     type=float, default=0.2,  help="PPO clip epsilon")
    parser.add_argument("--vf_coef",        type=float, default=0.5,  help="Value-head loss coefficient")
    parser.add_argument("--max_grad_norm",  type=float, default=1.0)
    parser.add_argument("--num_samples",    type=int,   default=-1,   help="-1 = full dataset")
    args = parser.parse_args()
    main(args)
