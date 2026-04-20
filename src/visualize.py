"""
Visualization of training curves and evaluation results across all alignment methods.

Reads data from:
  results/{method}/trainer_state.json  – TRL trainer log  (DPO, KTO, Online-DPO, RLOO)
  results/{method}/train_log.json      – Custom PPO log
  results/{method}/eval_results.json   – Evaluation accuracy (all methods)

Produces a single figure (results/comparison.png) with four panels:

  ┌───────────────────────┬───────────────────────┐
  │  Training Loss        │  RL Mean Reward       │
  │  (all methods)        │  (PPO + RLOO)         │
  ├───────────────────────┼───────────────────────┤
  │  Eval Accuracy        │  Final Accuracy       │
  │  per epoch (TRL)      │  Comparison (bar-line)│
  └───────────────────────┴───────────────────────┘

Usage
-----
    python visualize.py [--results_dir ../results] [--out comparison.png]

If result files are not yet present the script generates synthetic demo data
so the figure layout can be reviewed before training.
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")   # headless rendering (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

METHODS = ["PPO", "DPO", "Online-DPO", "KTO", "RLOO"]

# Map method name → results sub-directory
METHOD_DIRS = {
    "PPO":        "ppo",
    "DPO":        "dpo",
    "Online-DPO": "online_dpo",
    "KTO":        "kto",
    "RLOO":       "rloo",
}

# Colours consistent across all panels
PALETTE = {
    "PPO":        "#e6194b",   # red
    "DPO":        "#3cb44b",   # green
    "Online-DPO": "#4363d8",   # blue
    "KTO":        "#f58231",   # orange
    "RLOO":       "#911eb4",   # purple
}

LINE_STYLES = {
    "PPO":        "-",
    "DPO":        "--",
    "Online-DPO": "-.",
    "KTO":        ":",
    "RLOO":       (0, (3, 1, 1, 1)),   # dash-dot-dot
}

MARKERS = {
    "PPO":        "o",
    "DPO":        "s",
    "Online-DPO": "^",
    "KTO":        "D",
    "RLOO":       "P",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict | None:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_train_log(results_dir: str, method: str) -> list[dict]:
    """
    Return list of {"step", "epoch", "loss", ...} dicts.
    Checks both trainer_state.json (TRL) and train_log.json (custom PPO).
    """
    base = os.path.join(results_dir, METHOD_DIRS[method])

    # TRL saves trainer_state.json
    state = _load_json(os.path.join(base, "trainer_state.json"))
    if state:
        return state.get("log_history", [])

    # Custom PPO saves train_log.json
    custom = _load_json(os.path.join(base, "train_log.json"))
    if custom:
        return custom.get("log_history", [])

    return []


def load_eval_results(results_dir: str, method: str) -> dict | None:
    base = os.path.join(results_dir, METHOD_DIRS[method])
    return _load_json(os.path.join(base, "eval_results.json"))


# ── Synthetic demo data ───────────────────────────────────────────────────────

def _synthetic_loss(steps: np.ndarray, start: float, noise_std: float = 0.05) -> np.ndarray:
    """Exponential decay + noise."""
    rng = np.random.default_rng(abs(hash(str(start))) % (2**31))
    decay = start * np.exp(-steps / (steps[-1] * 0.6))
    return np.clip(decay + rng.normal(0, noise_std, len(steps)), 0.05, None)


def _synthetic_reward(steps: np.ndarray, plateau: float) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(str(plateau))) % (2**31))
    rise = plateau * (1 - np.exp(-steps / (steps[-1] * 0.4)))
    return rise + rng.normal(0, 0.02, len(steps))


DEMO_FINAL_ACC = {
    "PPO":        0.62,
    "DPO":        0.58,
    "Online-DPO": 0.61,
    "KTO":        0.55,
    "RLOO":       0.63,
}

DEMO_LOSS_START = {
    "PPO":        1.80,
    "DPO":        1.60,
    "Online-DPO": 1.65,
    "KTO":        1.70,
    "RLOO":       1.75,
}

DEMO_REWARD_PLATEAU = {
    "PPO":  0.45,
    "RLOO": 0.52,
}


def build_demo_data(n_steps: int = 200) -> dict:
    """Return synthetic training data shaped like real loaded data."""
    steps = np.linspace(1, n_steps, n_steps)
    epochs = np.linspace(0, 1, n_steps)

    demo: dict = {"train_logs": {}, "eval_results": {}}
    eval_points = np.linspace(0, 1, 5)   # 5 eval checkpoints per epoch
    for m in METHODS:
        losses = _synthetic_loss(steps, DEMO_LOSS_START[m])
        log = [
            {"step": int(s), "epoch": float(e), "loss": float(l)}
            for s, e, l in zip(steps, epochs, losses)
        ]
        if m in DEMO_REWARD_PLATEAU:
            rewards = _synthetic_reward(steps, DEMO_REWARD_PLATEAU[m])
            for entry, r in zip(log, rewards):
                entry["mean_reward"] = float(r)
        # Add synthetic eval_loss checkpoints
        rng = np.random.default_rng(abs(hash(m)) % (2**31))
        for ep in eval_points:
            log.append({
                "step":      int(ep * n_steps),
                "epoch":     float(ep),
                "eval_loss": float(DEMO_LOSS_START[m] * np.exp(-ep * 1.2) + rng.normal(0, 0.03)),
            })
        demo["train_logs"][m] = log
        demo["eval_results"][m] = {
            "method":   m,
            "accuracy": DEMO_FINAL_ACC[m],
            "correct":  int(DEMO_FINAL_ACC[m] * 1319),
            "num_samples": 1319,
        }

    return demo


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _smooth(values: list[float], window: int = 15) -> np.ndarray:
    """Uniform moving-average smoothing."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Pad front so length matches
    pad = np.full(len(arr) - len(smoothed), smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_training_loss(ax: plt.Axes, train_logs: dict[str, list[dict]], is_demo: bool) -> None:
    ax.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    plotted = False
    for method in METHODS:
        log = train_logs.get(method, [])
        loss_entries = [e for e in log if "loss" in e]
        if not loss_entries:
            continue
        steps  = [e["step"] for e in loss_entries]
        losses = [e["loss"] for e in loss_entries]
        smooth = _smooth(losses)
        ax.plot(
            steps, smooth,
            color=PALETTE[method],
            linestyle=LINE_STYLES[method],
            linewidth=2,
            label=method,
        )
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No training data found", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    if is_demo:
        ax.text(0.02, 0.98, "demo data", transform=ax.transAxes,
                fontsize=8, color="gray", va="top")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


def plot_rl_reward(ax: plt.Axes, train_logs: dict[str, list[dict]], is_demo: bool) -> None:
    ax.set_title("RL Mean Reward (PPO / RLOO)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")

    rl_methods = ["PPO", "RLOO"]
    plotted = False
    for method in rl_methods:
        log = train_logs.get(method, [])
        reward_entries = [e for e in log if "mean_reward" in e]
        if not reward_entries:
            continue
        steps   = [e["step"] for e in reward_entries]
        rewards = [e["mean_reward"] for e in reward_entries]
        smooth  = _smooth(rewards)
        ax.plot(
            steps, smooth,
            color=PALETTE[method],
            linestyle=LINE_STYLES[method],
            linewidth=2.5,
            label=method,
        )
        ax.fill_between(
            steps, np.array(rewards) - 0.05, np.array(rewards) + 0.05,
            alpha=0.12, color=PALETTE[method],
        )
        plotted = True

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="reward = 0")
    if not plotted:
        ax.text(0.5, 0.5, "No reward data found", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    if is_demo:
        ax.text(0.02, 0.98, "demo data", transform=ax.transAxes,
                fontsize=8, color="gray", va="top")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_eval_accuracy_per_epoch(
    ax: plt.Axes,
    train_logs: dict[str, list[dict]],
    is_demo: bool,
) -> None:
    """
    Plot eval accuracy sampled from TRL's log_history entries that contain
    'eval_accuracy' or 'eval_loss'.  Falls back to plotting eval_loss (inverted
    scale, right y-axis) when accuracy is unavailable.
    """
    ax.set_title("Eval Loss over Training", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Eval Loss")

    plotted = False
    for method in METHODS:
        log = train_logs.get(method, [])
        # Prefer eval_accuracy; fall back to eval_loss
        acc_entries = [e for e in log if "eval_accuracy" in e]
        loss_entries = [e for e in log if "eval_loss" in e]

        if acc_entries:
            ax.set_ylabel("Eval Accuracy")
            epochs = [e.get("epoch", e.get("step", i)) for i, e in enumerate(acc_entries)]
            values = [e["eval_accuracy"] for e in acc_entries]
            ax.plot(epochs, values, color=PALETTE[method],
                    linestyle=LINE_STYLES[method], linewidth=2,
                    marker=MARKERS[method], markersize=5, label=method)
            plotted = True

        elif loss_entries:
            epochs = [e.get("epoch", e.get("step", i)) for i, e in enumerate(loss_entries)]
            values = [e["eval_loss"] for e in loss_entries]
            ax.plot(epochs, values, color=PALETTE[method],
                    linestyle=LINE_STYLES[method], linewidth=2,
                    marker=MARKERS[method], markersize=5, label=method)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No eval data found\n(run evaluate.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray",
                fontsize=10)
    if is_demo:
        ax.text(0.02, 0.98, "demo data", transform=ax.transAxes,
                fontsize=8, color="gray", va="top")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_final_accuracy(
    ax: plt.Axes,
    eval_results: dict[str, dict | None],
    baseline_acc: float | None,
    is_demo: bool,
) -> None:
    """Bar-style line chart: one marker per method connected by a line."""
    ax.set_title("Final Accuracy on GSM8k Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy (%)")

    methods_present = [m for m in METHODS if eval_results.get(m)]
    accs = [eval_results[m]["accuracy"] * 100 for m in methods_present]

    if not methods_present:
        ax.text(0.5, 0.5, "No eval_results.json found\n(run evaluate.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray",
                fontsize=10)
        if is_demo:
            ax.text(0.02, 0.98, "demo data", transform=ax.transAxes,
                    fontsize=8, color="gray", va="top")
        return

    x = np.arange(len(methods_present))

    # Connecting line
    ax.plot(x, accs, color="dimgray", linewidth=1.5, zorder=1, alpha=0.6)

    # Individual markers per method
    for i, (method, acc) in enumerate(zip(methods_present, accs)):
        ax.scatter(
            [i], [acc],
            color=PALETTE[method],
            s=120,
            zorder=3,
            marker=MARKERS[method],
            edgecolors="white",
            linewidths=1.2,
            label=f"{method} ({acc:.1f}%)",
        )
        ax.annotate(
            f"{acc:.1f}%",
            xy=(i, acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=PALETTE[method],
            fontweight="bold",
        )

    # Baseline reference line
    if baseline_acc is not None:
        ax.axhline(
            baseline_acc * 100,
            color="black", linewidth=1.2, linestyle="--", alpha=0.7,
            label=f"Baseline ({baseline_acc*100:.1f}%)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods_present, fontsize=10)
    ax.set_ylim(bottom=max(0, min(accs) - 10), top=min(100, max(accs) + 10))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    if is_demo:
        ax.text(0.02, 0.98, "demo data", transform=ax.transAxes,
                fontsize=8, color="gray", va="top")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_dir = os.path.abspath(args.results_dir)
    out_path    = os.path.join(results_dir, args.out)

    # ── Load real data ────────────────────────────────────────────────────────
    train_logs:   dict[str, list[dict]] = {}
    eval_results: dict[str, dict | None] = {}

    for method in METHODS:
        train_logs[method]   = load_train_log(results_dir, method)
        eval_results[method] = load_eval_results(results_dir, method)

    baseline_path = os.path.join(results_dir, "baseline", "eval_results.json")
    baseline_data = _load_json(baseline_path)
    baseline_acc  = baseline_data["accuracy"] if baseline_data else None

    # Detect if we have any real data at all
    has_train = any(len(v) > 0 for v in train_logs.values())
    has_eval  = any(v is not None for v in eval_results.values())
    is_demo   = not (has_train or has_eval)

    if is_demo:
        warnings.warn(
            "No result files found – rendering with synthetic demo data. "
            "Run training scripts and evaluate.py first for real plots.",
            stacklevel=2,
        )
        demo = build_demo_data()
        train_logs   = demo["train_logs"]
        eval_results = demo["eval_results"]
        baseline_acc = 0.42   # plausible GSM8k 0-shot baseline for 1b model
    elif not has_train:
        warnings.warn("No training logs found; training-curve panels will be empty.")
    elif not has_eval:
        warnings.warn("No eval_results.json found; accuracy panel will be empty.")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "LLM Alignment Methods on GSM8k – Training & Evaluation Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    plot_training_loss(          axes[0, 0], train_logs,   is_demo)
    plot_rl_reward(              axes[0, 1], train_logs,   is_demo)
    plot_eval_accuracy_per_epoch(axes[1, 0], train_logs,   is_demo)
    plot_final_accuracy(         axes[1, 1], eval_results, baseline_acc, is_demo)

    # ── Global legend note ────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0],
                   color=PALETTE[m], linestyle=LINE_STYLES[m],
                   linewidth=2, marker=MARKERS[m], markersize=7, label=m)
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(METHODS),
        fontsize=10,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize alignment training results")
    parser.add_argument(
        "--results_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Root directory containing per-method result sub-folders",
    )
    parser.add_argument(
        "--out", type=str, default="comparison.png",
        help="Output filename (saved inside --results_dir)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Call plt.show() after saving (requires a display)",
    )
    args = parser.parse_args()
    main(args)
