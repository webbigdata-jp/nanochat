# scripts/chat_cli_jp.py

import argparse
import torch
from nanochat.common import compute_init, get_base_dir
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model_from_dir
import os

def main():
    parser = argparse.ArgumentParser(description='Chat with the Japanese model')
    parser.add_argument('-i', '--source', type=str, default="sft_jp")
    parser.add_argument('-p', '--prompt', type=str, default='')
    parser.add_argument('-t', '--temperature', type=float, default=0.6)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-m', '--max-tokens', type=int, default=128)
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # 日本語モデル用のディレクトリからロード
    base_dir = get_base_dir()
    source_map = {"sft_jp": "chatsft_checkpoints_jp"}
    checkpoints_dir = os.path.join(base_dir, source_map[args.source])
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="eval")
    engine = Engine(model, tokenizer)

    # 特殊トークンの準備
    bos, user_start, user_end = tokenizer.get_bos_token_id(), tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]

    while True:
        user_input = args.prompt if args.prompt else input("\nあなた: ").strip()
        if not user_input: continue

        conversation_tokens.extend([user_start] + tokenizer.encode(user_input) + [user_end, assistant_start])
        
        print("\nモデル: ", end="", flush=True)
        with autocast_ctx:
            for token_column, _ in engine.generate(conversation_tokens, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k):
                token = token_column[0]
                if token == assistant_end: break
                print(tokenizer.decode([token]), end="", flush=True)
                conversation_tokens.append(token)
        
        conversation_tokens.append(assistant_end)
        print()

        if args.prompt: break # プロンプトモードなら1回で終了
        if user_input.lower() in ['quit', 'exit']: break

if __name__ == "__main__":
    main()

