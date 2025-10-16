import sys
import os
sys.path.append(os.getcwd()) # nanochatのパッケージをインポートするために必要

import requests
import pyarrow.ipc
import torch
import time
from typing import List, Dict, Any

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# --- 設定項目 ---

# 1. データセット設定
DATASET_URLS = [
    "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00000-of-00002.arrow",
    "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00001-of-00002.arrow"
]

# 2. モデル設定
TOKENIZER_PATH = "japanese_tokenizer"
MODEL_CONFIG = { "sequence_len": 1024, "vocab_size": 65536, "n_layer": 12, "n_head": 6, "n_kv_head": 6, "n_embd": 768 }

# 3. SFT学習設定
NUM_TRAINING_STEPS = 40
DEVICE_BATCH_SIZE = 2
LEARNING_RATE = 1e-4
LOG_INTERVAL = 5

# --- データセット処理 ---

def download_and_load_data(urls: List[str]) -> List[Dict[str, Any]]:
    all_data = []
    for i, url in enumerate(urls):
        print(f"Downloading data from {url} ({i+1}/{len(urls)})...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with pyarrow.ipc.open_stream(response.content) as reader:
                for batch in reader:
                    all_data.extend(batch.to_pylist())
            print(f"Successfully loaded {len(all_data)} records so far.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data from {url}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing Arrow file from {url}: {e}")
            sys.exit(1)
    return all_data

class JpFlanTask:
    def __init__(self):
        self.data = download_and_load_data(DATASET_URLS)
        print(f"\nTotal {len(self.data):,} records loaded from Ego/jpflan dataset.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        user_content = item['input'].replace("[INST]", "").replace("[/INST]", "").strip()
        assistant_content = item['output']
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }

# --- データローダ ---

def sft_data_generator(dataset, tokenizer, batch_size, device):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    batch = []
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            
            if len(batch) == batch_size:
                max_len = max(len(ids) for ids, mask in batch)
                inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
                targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
                
                for j, (ids, mask) in enumerate(batch):
                    seq_len = len(ids)
                    inputs[j, :seq_len-1] = torch.tensor(ids[:-1])
                    row_targets = torch.tensor(ids[1:])
                    mask_tensor = torch.tensor(mask[1:])
                    row_targets[mask_tensor == 0] = -1
                    targets[j, :seq_len-1] = row_targets

                yield inputs.to(device), targets.to(device)
                batch = []

# --- メインの実行部分 ---
if __name__ == "__main__":
    print("--- Starting Japanese SFT Test Script ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ★★★★★ 修正点1: autocastコンテキストを準備 ★★★★★
    use_bf16 = (device == 'cuda' and torch.cuda.is_bf16_supported())
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16)
    if use_bf16:
        print("bfloat16 is supported. Using torch.autocast for mixed precision.")

    # 1. トークナイザーをロード
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at '{TOKENIZER_PATH}'. Please run 'test_japanese_tokenizer.py' first.")
        sys.exit(1)
    print(f"Loading tokenizer from '{TOKENIZER_PATH}'...")
    tokenizer = RustBPETokenizer.from_directory(TOKENIZER_PATH)

    # 2. モデルを初期化
    print("Initializing GPT model...")
    config = GPTConfig(**MODEL_CONFIG)
    model = GPT(config)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # 3. オプティマイザをセットアップ
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. データセットとデータローダを準備
    print("\nPreparing JpFlan dataset...")
    jpflan_dataset = JpFlanTask()
    train_loader = sft_data_generator(jpflan_dataset, tokenizer, DEVICE_BATCH_SIZE, device)
    
    # 5. 学習ループ
    print(f"\nStarting SFT for {NUM_TRAINING_STEPS} steps...")
    model.train()
    for step in range(NUM_TRAINING_STEPS):
        t0 = time.time()
        
        inputs, targets = next(train_loader)
        
        # ★★★★★ 修正点2: フォワードパスをautocastで囲む ★★★★★
        with autocast_ctx:
            loss = model(inputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t1 = time.time()
        if step % LOG_INTERVAL == 0 or step == NUM_TRAINING_STEPS - 1:
            print(f"Step {step:4d}/{NUM_TRAINING_STEPS} | Loss: {loss.item():.4f} | Time: {(t1 - t0) * 1000:.2f}ms")

    print("\n--- SFT test finished successfully! ---")

