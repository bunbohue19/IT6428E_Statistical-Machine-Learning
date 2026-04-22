"""
Visualize training dynamics across alignment methods (DPO, KTO, PPO, RLOO, Online-DPO).

For each method directory under ``results/`` that contains a HuggingFace
``checkpoint-*/trainer_state.json``, this script extracts the training log
history and plots comparable metrics side-by-side.

Usage
-----
    python visualize.py [--results_dir ../results] [--out comparison.png]
                        [--methods dpo kto] [--smooth 10]
"""

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "dpo":        "DPO",
    "kto":        "KTO",
    "ppo":        "PPO",
    "rloo":       "RLOO",
    "online_dpo": "Online-DPO",
}

METHOD_COLORS = {
    "dpo":        "#1f77b4",
    "kto":        "#ff7f0e",
    "ppo":        "#2ca02c",
    "rloo":       "#d62728",
    "online_dpo": "#9467bd",
}


def find_trainer_state(method_dir: str) -> str | None:
    """Return the trainer_state.json path from the latest checkpoint in method_dir."""
    ckpts = glob(os.path.join(method_dir, "checkpoint-*"))
    if not ckpts:
        return None
    # pick highest step
    ckpts.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
    path = os.path.join(ckpts[-1], "trainer_state.json")
    return path if os.path.isfile(path) else None


def load_log_history(method_dir: str) -> list[dict]:
    path = find_trainer_state(method_dir)
    if path is None:
        return []
    with open(path) as f:
        state = json.load(f)
    return state.get("log_history", [])


def load_eval_accuracy(method_dir: str) -> float | None:
    path = os.path.join(method_dir, "eval_results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get("accuracy")


def extract(logs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for entries that contain *key*."""
    xs, ys = [], []
    for row in logs:
        if key in row and "step" in row:
            xs.append(row["step"])
            ys.append(row[key])
    return np.asarray(xs), np.asarray(ys, dtype=float)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_metric(ax, runs: dict, key: str, title: str, window: int,
                ylabel: str | None = None):
    """Plot *key* from each run on *ax*. Skip runs that don't log that key."""
    any_plotted = False
    for method, logs in runs.items():
        xs, ys = extract(logs, key)
        if len(xs) == 0:
            continue
        ys_s = smooth(ys, window)
        xs_s = xs[len(xs) - len(ys_s):]
        ax.plot(xs_s, ys_s,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                linewidth=1.6)
        any_plotted = True
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel or key)
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray")


def plot_chosen_vs_rejected(ax, runs: dict, window: int):
    """Overlay rewards/chosen (solid) and rewards/rejected (dashed) per method."""
    any_plotted = False
    for method, logs in runs.items():
        xs_c, ys_c = extract(logs, "rewards/chosen")
        xs_r, ys_r = extract(logs, "rewards/rejected")
        color = METHOD_COLORS.get(method)
        label = METHOD_LABELS.get(method, method)
        if len(xs_c):
            y = smooth(ys_c, window)
            ax.plot(xs_c[len(xs_c) - len(y):], y,
                    color=color, linewidth=1.6, label=f"{label} chosen")
            any_plotted = True
        if len(xs_r):
            y = smooth(ys_r, window)
            ax.plot(xs_r[len(xs_r) - len(y):], y,
                    color=color, linewidth=1.2, linestyle="--",
                    label=f"{label} rejected")
            any_plotted = True
    ax.set_title("Implicit rewards: chosen vs rejected")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    if any_plotted:
        ax.legend(fontsize=7)


def plot_accuracy_bar(ax, accuracies: dict):
    if not accuracies:
        ax.text(0.5, 0.5, "no eval_results.json found",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Final GSM8k accuracy")
        ax.set_axis_off()
        return
    methods = list(accuracies.keys())
    values  = [accuracies[m] * 100 for m in methods]
    colors  = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Final GSM8k accuracy")
    ax.set_ylabel("accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="../results")
    parser.add_argument("--out",         default="../results/comparison.png")
    parser.add_argument("--methods",     nargs="*", default=None,
                        help="Subset of methods to plot (default: all found).")
    parser.add_argument("--smooth",      type=int, default=5,
                        help="Moving-average window over training steps.")
    args = parser.parse_args()

    # Discover method dirs
    if args.methods:
        method_names = args.methods
    else:
        method_names = sorted(
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        )

    runs: dict[str, list[dict]] = {}
    accuracies: dict[str, float] = {}
    for name in method_names:
        method_dir = os.path.join(args.results_dir, name)
        logs = load_log_history(method_dir)
        acc  = load_eval_accuracy(method_dir)
        if logs:
            runs[name] = logs
            print(f"  [ok]   {name}: {len(logs)} log entries")
        elif acc is None:
            print(f"  [skip] {name}: no trainer_state.json and no eval_results.json")
            continue
        else:
            print(f"  [eval] {name}: no training logs, eval-only")
        if acc is not None:
            accuracies[name] = acc

    if not runs and not accuracies:
        raise SystemExit(f"No training runs or eval results found under {args.results_dir}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_metric(axes[0, 0], runs, "loss",
                "Training loss", args.smooth, ylabel="loss")
    plot_metric(axes[0, 1], runs, "rewards/margins",
                "Reward margin (chosen − rejected)", args.smooth,
                ylabel="margin")
    plot_chosen_vs_rejected(axes[0, 2], runs, args.smooth)
    plot_metric(axes[1, 0], runs, "rewards/accuracies",
                "Preference accuracy", args.smooth,
                ylabel="accuracy")
    plot_metric(axes[1, 1], runs, "kl",
                "KL divergence to reference", args.smooth,
                ylabel="KL")
    plot_accuracy_bar(axes[1, 2], accuracies)

    fig.suptitle("Alignment training comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {args.out}")


if __name__ == "__main__":
    main()
