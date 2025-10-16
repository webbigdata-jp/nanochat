# scripts/base_train_jp.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
from collections import deque

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, get_dist_info
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb

from nanochat.dataset_jp import direct_download_iterator

# --- 設定項目 ---
DATASET_REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
TOTAL_SHARDS = 378
DOWNLOAD_CACHE_DIR = "download_cache_jp"

depth = 12
max_seq_len = 1024
num_iterations = 100
device_batch_size = 4
total_batch_size = 262144
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
eval_every = 50
eval_tokens = 10 * 524288

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# --- データローダ ---

# ★変更点1: ddp_rankとddp_world_sizeを引数で受け取るように変更
def get_jp_pretrain_batch_iterator(tokenizer, batch_size, context_length, device, ddp_rank, ddp_world_size):
    """
    日本語データセット用の事前学習データジェネレータ。
    """
    text_iterator = direct_download_iterator(
        DATASET_REPO_ID, CONFIG_NAME, SPLIT, TOTAL_SHARDS, DOWNLOAD_CACHE_DIR,
        start=ddp_rank, step=ddp_world_size
    )

    token_buffer = deque()
    while True:
        for text_batch in text_iterator:
            token_ids_list = tokenizer.encode(text_batch, num_threads=8)
            for ids in token_ids_list:
                token_buffer.extend(ids)

            while len(token_buffer) >= batch_size * context_length + 1:
                x_chunks, y_chunks = [], []
                for _ in range(batch_size):
                    end_idx = context_length
                    # Deque to list conversion can be slow, let's optimize
                    chunk = [token_buffer.popleft() for _ in range(end_idx)]
                    x_chunks.append(chunk)
                    y_chunks.append(chunk[1:] + [token_buffer[0]]) # Correctly form y
                
                X = torch.tensor(x_chunks, dtype=torch.long, device=device)
                Y = torch.tensor(y_chunks, dtype=torch.long, device=device)
                yield X, Y

# --- メイン実行部 ---

def main():
    # ★変更点2: ここで一度だけcompute_init()を呼び出す
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Japanese Tokenizer loaded. Vocab size: {vocab_size:,}")

    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim)
    with torch.device("meta"):
        model = GPT(GPTConfig(**model_config_kwargs))
    model.to_empty(device="cuda")
    model.init_weights()
    orig_model = model
    model = torch.compile(model, dynamic=False)
    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Model initialized with {num_params/1e6:.2f}M parameters.")

    optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)

    # ★変更点3: ddp_rankとddp_world_sizeをデータローダに渡す
    train_loader = get_jp_pretrain_batch_iterator(tokenizer, device_batch_size, max_seq_len, device, ddp_rank, ddp_world_size)
    build_val_loader = lambda: get_jp_pretrain_batch_iterator(tokenizer, device_batch_size, max_seq_len, device, ddp_rank, ddp_world_size)
    x, y = next(train_loader)

    tokens_per_fwdbwd = device_batch_size * max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    print0(f"Gradient accumulation steps: {grad_accum_steps}")

    for step in range(num_iterations + 1):
        last_step = step == num_iterations

        if last_step or step % eval_every == 0:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = eval_tokens // world_tokens_per_fwdbwd
            with autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            model.train()

        if master_process and last_step:
            output_dirname = f"d{depth}"
            checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints_jp", output_dirname)
            save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), None, { "step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs })
            print0(f"Saved Japanese model checkpoint to {checkpoint_dir}")

        if last_step:
            break

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)
        
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % 10 == 0:
            print0(f"Step {step:05d}/{num_iterations:05d} | loss: {loss.item() * grad_accum_steps:.6f}")

    compute_cleanup()

if __name__ == "__main__":
    main()

