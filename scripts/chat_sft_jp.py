
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, get_dist_info
from nanochat.checkpoint_manager import load_model_from_dir, save_checkpoint

# ★変更点: 日本語SFTデータ用のタスククラスをインポート
from tasks.jpflan import JpFlan

# --- 設定項目 ---
source = "base_jp" # base_jp | mid_jp などを想定
model_tag = None
step = None
num_iterations = 100 # テスト用に短いステップ数
device_batch_size = 1
target_examples_per_step = 4 # 勾配累積をテスト
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
eval_every = 50

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# --- データローダ (sft_jp_test.pyから移植・DDP対応) ---

def sft_data_generator(dataset, tokenizer, batch_size, ddp_rank, ddp_world_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        # ... (sft_jp_test.pyと同じロジック)
        nrows = len(batch)
        max_len = max(len(ids) for ids, mask in batch)
        inputs = torch.full((nrows, max_len - 1), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, max_len - 1), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)

    batch = []
    while True:
        # DDP対応: 各プロセスが担当するインデックスを処理
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# --- メイン実行部 ---

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# ★変更点: 日本語モデル用のディレクトリからロード
base_dir = get_base_dir()
source_map = {
    "base_jp": "base_checkpoints_jp",
    "mid_jp": "mid_checkpoints_jp", # 今後中間学習を実装した場合用
}
checkpoints_dir = os.path.join(base_dir, source_map[source])
model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="train", model_tag=model_tag, step=step)

# データセット準備
train_ds = JpFlan(split="train")
train_loader = sft_data_generator(train_ds, tokenizer, device_batch_size, ddp_rank, ddp_world_size)

# オプティマイザ準備
optimizers = model.setup_optimizers(unembedding_lr, embedding_lr, matrix_lr, weight_decay)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] *= init_lr_frac
        group["initial_lr"] = group["lr"]

# 勾配累積ステップ数の計算
examples_per_step = device_batch_size * ddp_world_size
grad_accum_steps = target_examples_per_step // examples_per_step

# 学習ループ
print0(f"Starting SFT for {num_iterations} iterations...")
for step in range(num_iterations):
    model.train()
    
    # 勾配累積ループ
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        loss = loss / grad_accum_steps
        loss.backward()

    # パラメータ更新
    for opt in optimizers: opt.step()
    model.zero_grad(set_to_none=True)

    if step % 10 == 0 or step == num_iterations - 1:
        print0(f"Step {step:05d}/{num_iterations:05d} | loss: {loss.item() * grad_accum_steps:.6f}")

# チェックポイント保存
if master_process:
    depth = model.config.n_layer
    output_dirname = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints_jp", output_dirname)
    save_checkpoint(checkpoint_dir, step, model.state_dict(), None, { "step": step, "model_config": model.config.__dict__ })
    print0(f"Saved Japanese SFT model checkpoint to {checkpoint_dir}")

compute_cleanup()

