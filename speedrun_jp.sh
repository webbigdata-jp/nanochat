#!/bin/bash
set -e # exit on first error

# ------------------------------------------------------------------------------------------
# Part 1, 2, 3 (変更なし)
# ------------------------------------------------------------------------------------------
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [ -f "$HOME/.cargo/env" ]; then source "$HOME/.cargo/env"; fi
pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
if ! command -v cargo &> /dev/null; then echo "Rust compiler 'cargo' not found."; exit; fi

python -m scripts.tok_train_jp --max_chars=1000000
python -m scripts.tok_eval

echo "Starting Japanese pre-training..."
torchrun --standalone --nproc_per_node=1 -m scripts.base_train_jp -- \
    --num_iterations=100 \
    --device_batch_size=2

# ------------------------------------------------------------------------------------------
# Part 4: SFT (Supervised Fine-Tuning) (変更なし)
# ------------------------------------------------------------------------------------------
echo "Starting Japanese SFT..."
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft_jp -- \
    --source=base_jp \
    --num_iterations=100 \
    --device_batch_size=1 \
    --target_examples_per_step=4

# ------------------------------------------------------------------------------------------
# Part 5: Evaluation (★ここから追加・変更)
# ------------------------------------------------------------------------------------------
echo "SFT complete. Starting evaluation..."

# 1. 定量的評価 (パイプラインの動作確認)
echo -e "\n--- Running Quantitative Evaluation (ARC-Easy Benchmark) ---"
python -m scripts.chat_eval_jp --source=sft_jp --task-name=ARC-Easy

# 2. 定性的評価 (実際の対話)
echo -e "\n--- Running Qualitative Evaluation (Sample Prompts) ---"
PROMPTS=(
    "日本の首都はどこですか？"
    "空はどうして青いのですか？"
    "1たす1は？"
)

for prompt in "${PROMPTS[@]}"; do
    echo "--------------------------------------------------"
    echo "あなた: $prompt"
    python -m scripts.chat_cli_jp --source=sft_jp --prompt="$prompt"
done
echo "--------------------------------------------------"

echo -e "\nJapanese model training and evaluation pipeline finished successfully!"


