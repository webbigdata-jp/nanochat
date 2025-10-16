import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import wandb
from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model_from_dir, save_checkpoint
from nanochat.report import get_report
from tasks.jpflan import JpFlan

# --- クラウド用設定 ---
run = "dummy"
source = "base_jp"
num_epochs = 1
max_iterations = -1
device_batch_size = 16
target_examples_per_step = 128 # 修正済みの値
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
eval_every = 100

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# --- データローダ ---
def sft_data_generator(dataset, tokenizer, batch_size, ddp_rank, ddp_world_size, device):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    def collate_and_yield(batch):
        nrows = len(batch)
        max_len = max(len(ids) for ids, mask in batch)
        inputs = torch.full((nrows, max_len - 1), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, max_len - 1), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids); ids_tensor = torch.tensor(ids, dtype=torch.long); inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]; mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1; targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]; ids, mask = tokenizer.render_conversation(doc); batch.append((ids, mask))
            if len(batch) == batch_size: yield collate_and_yield(batch); batch = []


# --- メイン実行部 ---
def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    use_dummy_wandb = run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-jp", name=run, config=user_config)

    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints_jp")
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, phase="train")
    
    train_ds = JpFlan(split="train")
    train_loader = sft_data_generator(train_ds, tokenizer, device_batch_size, ddp_rank, ddp_world_size, device)
    optimizers = model.setup_optimizers(unembedding_lr, embedding_lr, matrix_lr, weight_decay)
    for opt in optimizers:
        for group in opt.param_groups: group["lr"] *= init_lr_frac; group["initial_lr"] = group["lr"]

    if max_iterations < 0:
        num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
    else:
        num_iterations = max_iterations
    
    examples_per_step = device_batch_size * ddp_world_size
    grad_accum_steps = target_examples_per_step // examples_per_step
    
    print0(f"Starting SFT for {num_iterations} iterations with grad_accum_steps = {grad_accum_steps}...")
    final_loss = 0.0
    model.train() # ループの外で一度だけtrainモードに
    for step in range(num_iterations):
        
        for micro_step in range(grad_accum_steps):
            train_inputs, train_targets = next(train_loader)
            with autocast_ctx:
                loss = model(train_inputs, train_targets)
            loss = loss / grad_accum_steps
            loss.backward()

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == num_iterations - 1:
            lossf = loss.item() * grad_accum_steps # 表示用に元のスケールに戻す
            final_loss = lossf
            print0(f"Step {step:05d}/{num_iterations:05d} | loss: {lossf:.6f}")
            wandb_run.log({"step": step, "train/sft_loss": lossf})

    if master_process:
        depth = model.config.n_layer
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints_jp", f"d{depth}")
        save_checkpoint(checkpoint_dir, step, model.state_dict(), None, { "step": step, "model_config": model.config.__dict__ })

    get_report().log(section="Chat SFT Japanese", data=[user_config, {"Final training loss": final_loss}])
    wandb_run.finish()
    compute_cleanup()

if __name__ == "__main__":
    main()

