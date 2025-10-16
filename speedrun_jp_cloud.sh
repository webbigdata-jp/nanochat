#!/bin/bash
set -e # exit on first error
# This script runs the Japanese nanochat pipeline start to finish on a cloud instance.
# It is intended to run on a single 8XH100 node.

# ------------------------------------------------------------------------------------------
# Part 1: Housekeeping, virtual environment, dependencies
# ------------------------------------------------------------------------------------------
# (前回と同じ、環境設定の完全自動化版)
echo "--- Setting up prerequisites (Rust, uv)... ---"
if [ -f "$HOME/.cargo/env" ]; then source "$HOME/.cargo/env"; fi
if ! command -v cargo &> /dev/null; then
    echo "Rust (cargo) not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
echo "--- Prerequisites set up. Configuring Python environment... ---"
[ -d ".venv" ] || uv venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
if ! command -v cargo &> /dev/null || ! command -v uv &> /dev/null; then
    echo "FATAL: Could not find cargo or uv in PATH."
    exit 1
fi
echo "--- Python environment is ready. ---"
export WANDB_RUN="d20-jp-$(date +%s)"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY is not set. Using dummy logger."
    export WANDB_RUN="dummy"
fi
python -m nanochat.report reset

# ------------------------------------------------------------------------------------------
# Part 2: Tokenization
# ------------------------------------------------------------------------------------------
# トークナイザークラウドでおそすぎ問題
TOKENIZER_FILE="$HOME/.cache/nanochat/tokenizer/tokenizer.pkl"
if [ -f "$TOKENIZER_FILE" ]; then
    echo "Tokenizer already exists at $TOKENIZER_FILE. Skipping training."
else
    echo "Starting Tokenizer training..."
    python -m scripts.tok_train_jp --max_chars=75000000
fi
python -m scripts.tok_eval

# ------------------------------------------------------------------------------------------
# Part 2.5: Pre-download Japanese Dataset for Pre-training
# ------------------------------------------------------------------------------------------
echo "Pre-downloading Japanese dataset for pre-training..."
python -m scripts.download_jp_dataset -n 100 # d20モデル用に100シャードをダウンロード

# SFT用のデータセットも事前にダウンロード
echo "Pre-downloading Japanese dataset for SFT..."
python -m scripts.download_sft_jp_dataset

# ------------------------------------------------------------------------------------------
# Part 3: Base model (pretraining)
# ------------------------------------------------------------------------------------------
echo "Starting Japanese pre-training on 8xH100..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_train_jp_cloud -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=$WANDB_RUN

# ------------------------------------------------------------------------------------------
# Part 4: SFT (Supervised Fine-Tuning)
# ------------------------------------------------------------------------------------------
echo "Starting Japanese SFT on 8xH100..."
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft_jp_cloud -- \
    --num_epochs=1 \
    --device_batch_size=16 \
    --target_examples_per_step=128 \
    --run=$WANDB_RUN

# ------------------------------------------------------------------------------------------
# Part 5: Evaluation & Report
# ------------------------------------------------------------------------------------------
echo "SFT complete. Starting evaluation..."
# (評価スクリプトも前回作成したものをそのまま使用)
echo -e "\n--- Running Qualitative Evaluation (Sample Prompts) ---"
PROMPTS=(
    "日本の首都はどこですか？"
    "空はどうして青いのですか？"
    "nanochatについて説明してください。"
)
for prompt in "${PROMPTS[@]}"; do
    echo "--------------------------------------------------"
    echo "あなた: $prompt"
    python -m scripts.chat_cli_jp --source=sft_jp --prompt="$prompt"
done
echo "--------------------------------------------------"
python -m nanochat.report generate
echo -e "\nJapanese model training and evaluation pipeline finished successfully!"
