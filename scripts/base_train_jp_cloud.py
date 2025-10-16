import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
import wandb
from collections import deque

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.dataset_jp import direct_download_iterator

# --- クラウド用設定 (オリジナル準拠) ---
run = "dummy"
depth = 20
max_seq_len = 2048
target_param_data_ratio = 20
num_iterations = -1
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
matrix_lr = 0.02
weight_decay = 0.0
eval_every = 250
eval_tokens = 20 * 524288

DATASET_REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
TOTAL_SHARDS = 378
DOWNLOAD_CACHE_DIR = "download_cache_jp"

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

def get_jp_pretrain_batch_iterator(tokenizer, batch_size, context_length, device, ddp_rank, ddp_world_size):
    text_iterator = direct_download_iterator(
        DATASET_REPO_ID, CONFIG_NAME, SPLIT, TOTAL_SHARDS, DOWNLOAD_CACHE_DIR,
        start=ddp_rank, step=ddp_world_size
    )
    token_buffer = deque()
    while True:
        for text_batch in text_iterator:
            token_ids_list = tokenizer.encode(text_batch, num_threads=8)
            for ids in token_ids_list: token_buffer.extend(ids)
            while len(token_buffer) >= batch_size * context_length + 1:
                x_chunks, y_chunks = [], []
                for _ in range(batch_size):
                    chunk = [token_buffer.popleft() for _ in range(context_length)]
                    x_chunks.append(chunk)
                    y_chunks.append(chunk[1:] + [token_buffer[0]])
                yield torch.tensor(x_chunks, dtype=torch.long, device=device), torch.tensor(y_chunks, dtype=torch.long, device=device)

def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    use_dummy_wandb = run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-jp", name=run, config=user_config)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=tokenizer.get_vocab_size(), n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim)
    with torch.device("meta"): model = GPT(GPTConfig(**model_config_kwargs))
    model.to_empty(device="cuda")
    model.init_weights()
    orig_model = model
    model = torch.compile(model, dynamic=False)
    num_params = sum(p.numel() for p in model.parameters())
    num_flops_per_token = model.estimate_flops()
    print0(f"Model initialized with {num_params/1e6:.2f}M parameters.")

    global num_iterations
    if num_iterations <= 0:
        target_tokens = target_param_data_ratio * num_params
        num_iterations = int(target_tokens // total_batch_size)
    total_tokens = total_batch_size * num_iterations
    print0(f"Total training tokens: {total_tokens:,} ({num_iterations:,} iterations)")

    optimizers = model.setup_optimizers(unembedding_lr, embedding_lr, matrix_lr, weight_decay)
    train_loader = get_jp_pretrain_batch_iterator(tokenizer, device_batch_size, max_seq_len, device, ddp_rank, ddp_world_size)
    build_val_loader = lambda: get_jp_pretrain_batch_iterator(tokenizer, device_batch_size, max_seq_len, device, ddp_rank, ddp_world_size)
    x, y = next(train_loader)
    
    tokens_per_fwdbwd = device_batch_size * max_seq_len
    grad_accum_steps = total_batch_size // (tokens_per_fwdbwd * ddp_world_size)
    
    min_val_bpb = float("inf")
    for step in range(num_iterations + 1):
        last_step = step == num_iterations
        if last_step or step % eval_every == 0:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = eval_tokens // (tokens_per_fwdbwd * ddp_world_size)
            with autocast_ctx: val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            if val_bpb < min_val_bpb: min_val_bpb = val_bpb
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            wandb_run.log({"step": step, "val/bpb": val_bpb})
            model.train()
        
        if master_process and last_step:
            output_dirname = f"d{depth}"
            checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints_jp", output_dirname)
            save_checkpoint(checkpoint_dir, step, orig_model.state_dict(), None, { "step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs, "user_config": user_config })
        if last_step: break
        
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx: loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)
        for opt in optimizers: opt.step()
        model.zero_grad(set_to_none=True)
        dt = time.time() - t0
        
        if step % 10 == 0:
            lossf = loss.item() * grad_accum_steps
            print0(f"Step {step:05d}/{num_iterations:05d} | loss: {lossf:.6f} | dt: {dt*1000:.2f}ms")
            wandb_run.log({"step": step, "train/loss": lossf})
    
    # レポートへのログ記録
    get_report().log(section="Base model training Japanese", data=[
        user_config,
        {
            "Number of parameters": num_params,
            "Number of training tokens": total_tokens,
            "Minimum validation bpb": min_val_bpb,
            "Final validation bpb": val_bpb,
        }
    ])
    wandb_run.finish()
    compute_cleanup()

if __name__ == "__main__":
    main()

