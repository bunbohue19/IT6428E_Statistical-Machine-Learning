#!/usr/bin/env bash
# =============================================================================
# evaluate.sh – Run GSM8k evaluation for all trained alignment methods
#
# Usage:
#   bash evaluate.sh [--num_samples N] [--batch_size B] [--method METHOD]
#
# Options:
#   --num_samples N    Evaluate on first N test examples (default: all 1319)
#   --batch_size  B    Generation batch size (default: 8)
#   --method METHOD    Only evaluate this method (default: all)
#
# Requirements:
#   conda activate sml     (or whichever env has the deps installed)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"

export HF_HOME="$ROOT_DIR/huggingface"

# ── Defaults ─────────────────────────────────────────────────────────────────
NUM_SAMPLES=-1
BATCH_SIZE=8
ONLY_METHOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --method)      ONLY_METHOD="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Methods and their result directories ─────────────────────────────────────
declare -A METHOD_DIRS=(
    ["baseline"]="google/gemma-3-1b-it"
    ["PPO"]="$RESULTS_DIR/ppo"
    ["DPO"]="$RESULTS_DIR/dpo"
    ["Online-DPO"]="$RESULTS_DIR/online_dpo"
    ["KTO"]="$RESULTS_DIR/kto"
    ["RLOO"]="$RESULTS_DIR/rloo"
)

# ── Evaluation function ───────────────────────────────────────────────────────
run_eval() {
    local method="$1"
    local model_path="$2"

    echo ""
    echo "============================================================"
    echo " Evaluating: $method"
    echo " Model:      $model_path"
    echo "============================================================"

    # Skip if results directory doesn't exist (not yet trained)
    if [[ "$model_path" != "google/"* ]] && [[ ! -d "$model_path" ]]; then
        echo "  [SKIP] $model_path not found – train first."
        return
    fi

    python "$SCRIPT_DIR/evaluate.py" \
        --model_path    "$model_path" \
        --method        "$method"     \
        --num_samples   "$NUM_SAMPLES" \
        --batch_size    "$BATCH_SIZE"
}

# ── Summary helper ────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "============================================================"
    echo "  SUMMARY"
    echo "============================================================"
    printf "%-15s %-10s %-10s\n" "Method" "Accuracy" "Correct/Total"
    printf "%-15s %-10s %-10s\n" "------" "--------" "-------------"

    for method in baseline PPO DPO Online-DPO KTO RLOO; do
        local model_path="${METHOD_DIRS[$method]}"
        local results_file

        if [[ "$model_path" == "google/"* ]]; then
            results_file="$RESULTS_DIR/baseline/eval_results.json"
        else
            results_file="$model_path/eval_results.json"
        fi

        if [[ -f "$results_file" ]]; then
            acc=$(python -c "
import json
with open('$results_file') as f:
    d = json.load(f)
print(f\"{d['accuracy']*100:.2f}%  {d['correct']}/{d['num_samples']}\")
")
            printf "%-15s %s\n" "$method" "$acc"
        else
            printf "%-15s %-10s\n" "$method" "N/A"
        fi
    done
    echo ""
}

# ── Run evaluations ───────────────────────────────────────────────────────────
if [[ -n "$ONLY_METHOD" ]]; then
    if [[ -v "METHOD_DIRS[$ONLY_METHOD]" ]]; then
        run_eval "$ONLY_METHOD" "${METHOD_DIRS[$ONLY_METHOD]}"
    else
        echo "Unknown method '$ONLY_METHOD'. Available: ${!METHOD_DIRS[*]}"
        exit 1
    fi
else
    for method in baseline PPO DPO Online-DPO KTO RLOO; do
        run_eval "$method" "${METHOD_DIRS[$method]}"
    done
fi

# ── Print summary if all methods were run ─────────────────────────────────────
if [[ -z "$ONLY_METHOD" ]]; then
    print_summary
fi
