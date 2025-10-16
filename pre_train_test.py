import sys
import os
sys.path.append(os.getcwd())

import torch
import time
import requests
import urllib.parse
import pyarrow.parquet as pq

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# --- 設定項目 (変更なし) ---
MODEL_CONFIG = { "sequence_len": 1024, "vocab_size": 65536, "n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768 }
TRAINING_CONFIG = { "batch_size": 1, "num_steps": 200, "learning_rate": 1e-4, "log_interval": 10 }
DATASET_CONFIG = { "repo_id": "kajuma/ABEJA-CC-JA-edu", "config_name": "10%", "split": "train", "total_shards": 378 }
TOKENIZER_PATH = "japanese_tokenizer"
DOWNLOAD_CACHE_DIR = "download_cache"

# --- データ供給イテレータ (変更なし) ---
def direct_download_iterator(repo_id, config_name, split, total_shards, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    encoded_config = urllib.parse.quote(config_name)
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    for i in range(total_shards):
        filename = f"{split}-{i:05d}-of-{total_shards:05d}.parquet"
        url = f"{base_url}/{encoded_config}/{filename}"
        local_path = os.path.join(cache_dir, filename)
        print(f"\nDownloading: {url}")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 404: break
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024): f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}"); continue
        print(f"Processing {os.path.basename(local_path)}...")
        try:
            with pq.ParquetFile(local_path) as pf:
                for batch in pf.iter_batches(batch_size=8192):
                    yield batch.to_pydict()['content']
        finally:
            if os.path.exists(local_path): os.remove(local_path)

def get_pretrain_batch(text_iterator, tokenizer, batch_size, context_length, device):
    token_buffer = []
    for text_batch in text_iterator:
        token_ids_list = tokenizer.encode(text_batch, num_threads=8)
        for ids in token_ids_list: token_buffer.extend(ids)
        while len(token_buffer) >= batch_size * context_length + 1:
            x_chunks, y_chunks = [], []
            for _ in range(batch_size):
                end_idx = context_length
                x_chunks.append(token_buffer[:end_idx])
                y_chunks.append(token_buffer[1:end_idx+1])
                token_buffer = token_buffer[end_idx:]
            X = torch.tensor(x_chunks, dtype=torch.long, device=device)
            Y = torch.tensor(y_chunks, dtype=torch.long, device=device)
            yield X, Y

# --- メインの実行部分 ---
if __name__ == "__main__":
    print("--- Starting Japanese Pre-training Test (Final Clean Shutdown) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    use_bf16 = (device == 'cuda' and torch.cuda.is_bf16_supported())
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at '{TOKENIZER_PATH}'"); exit(1)
    print(f"Loading tokenizer from '{TOKENIZER_PATH}'...")
    tokenizer = RustBPETokenizer.from_directory(TOKENIZER_PATH)
    print("Initializing GPT model with original config...")
    config = GPTConfig(**MODEL_CONFIG)
    model = GPT(config)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
    if use_bf16:
        print("bfloat16 is supported. Using torch.autast for mixed precision.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    print("Setting up data loader...")
    text_iterator = direct_download_iterator(
        DATASET_CONFIG["repo_id"], DATASET_CONFIG["config_name"], DATASET_CONFIG["split"],
        DATASET_CONFIG["total_shards"], DOWNLOAD_CACHE_DIR
    )
    batch_iterator = get_pretrain_batch(
        text_iterator, tokenizer, TRAINING_CONFIG["batch_size"], MODEL_CONFIG["sequence_len"], device
    )
    
    # ★★★★★ ここが最後の、そして唯一の正しい修正点です ★★★★★
    # try...finallyブロックで学習ループを囲み、
    # 最後に必ずジェネレータをクローズして、リソースを安全に解放します。
    try:
        print(f"\nStarting training for {TRAINING_CONFIG['num_steps']} steps...")
        model.train()
        for step in range(TRAINING_CONFIG['num_steps']):
            t0 = time.time()
            try:
                X, Y = next(batch_iterator)
            except StopIteration:
                print("Data iterator exhausted. Test finished."); break
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            if step % TRAINING_CONFIG["log_interval"] == 0:
                print(f"Step {step:4d}/{TRAINING_CONFIG['num_steps']} | Loss: {loss.item():.4f} | Time: {(t1-t0)*1000:.2f}ms")
    finally:
        print("Closing data iterators...")
        batch_iterator.close()
        if 'text_iterator' in locals():
            try:
                text_iterator.close()
            except:
                pass


    print("\n--- Pre-training test completed successfully and cleanly! ---")

