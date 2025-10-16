#!/bin/bash
set -e # exit on first error

# ------------------------------------------------------------------------------------------
# Part 1: Housekeeping, virtual environment, dependencies
# ------------------------------------------------------------------------------------------
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [ -f "$HOME/.cargo/env" ]; then source "$HOME/.cargo/env"; fi
pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
if ! command -v cargo &> /dev/null; then echo "Rust compiler 'cargo' not found."; exit; fi

# ★追加: WandBのラン名を指定。クラウドで実行する際は "dummy" 以外に変更してください。
export WANDB_RUN="d20-jp-$(date +%s)"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY is not set. Using dummy logger."
    export WANDB_RUN="dummy"
fi

# ★追加: レポートシステムを初期化
python -m nanochat.report reset

# ------------------------------------------------------------------------------------------
# Part 2: Tokenization
# ------------------------------------------------------------------------------------------
python -m scripts.tok_train_jp --max_chars=2000000000
python -m scripts.tok_eval

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
    --target_examples_per_step=32 \
    --run=$WANDB_RUN

# ------------------------------------------------------------------------------------------
# Part 5: Evaluation & Report
# ------------------------------------------------------------------------------------------
echo "SFT complete. Starting evaluation..."
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

# ★追加: 最終的なレポートファイルを生成
python -m nanochat.report generate

echo -e "\nJapanese model training and evaluation pipeline finished successfully!"


