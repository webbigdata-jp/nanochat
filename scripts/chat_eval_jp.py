# scripts/chat_eval_jp.py

import argparse
import torch
from nanochat.common import compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval # 評価ロジック自体は再利用
import os
from nanochat.common import get_base_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', required=True, help="Source of the model: e.g., sft_jp")
    parser.add_argument('-a', '--task-name', default='ARC-Easy', help="Task name")
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('-s', '--step', type=int, default=None)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # 日本語モデル用のディレクトリからロード
    base_dir = get_base_dir()
    source_map = {
        "sft_jp": "chatsft_checkpoints_jp",
    }
    checkpoints_dir = os.path.join(base_dir, source_map[args.source])
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    
    print0(f"\n--- Running evaluation for task: {args.task_name} ---")
    print0("NOTE: This benchmark is in English. Scores are expected to be low for a Japanese model.")
    
    with autocast_ctx:
        acc = run_chat_eval(
            args.task_name, model, tokenizer, engine,
            batch_size=args.batch_size
        )
    print0(f"--- {args.task_name} accuracy: {100 * acc:.2f}% ---")
    compute_cleanup()

if __name__ == "__main__":
    main()


