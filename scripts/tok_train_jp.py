# scripts/tok_train_jp.py

import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
# ★変更点: 日本語データセット用のモジュールをインポート
from nanochat.dataset_jp import direct_download_iterator, limited_char_iterator

# --- 設定項目 (argparseで上書き可能) ---
DATASET_REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
TOTAL_SHARDS = 378 # 念のため最大値を設定しておくが、途中でループは止まるはず
DOWNLOAD_CACHE_DIR = "download_cache_jp" # 一時ダウンロードディレクトリ

def main():
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer for Japanese')
    # テストスクリプトの設定値をデフォルト値として採用
    parser.add_argument('--max_chars', type=int, default=500_000_000, help='Maximum characters to train on')
    parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size')
    args = parser.parse_args()
    print(f"max_chars: {args.max_chars:,}")
    print(f"vocab_size: {args.vocab_size:,}")

    # --- テキストイテレータの準備 ---
    base_iterator = direct_download_iterator(
        DATASET_REPO_ID, CONFIG_NAME, SPLIT, TOTAL_SHARDS, DOWNLOAD_CACHE_DIR
    )
    final_iterator = limited_char_iterator(base_iterator, args.max_chars)
    
    # --- トークナイザーの学習 ---
    print(f"\nStarting tokenizer training with vocab size: {args.vocab_size}...")
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(final_iterator, args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"\nTraining time: {train_time:.2f}s")

    # --- トークナイザーの保存 ---
    # ★重要: nanochatの標準ディレクトリに保存する
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)
    print(f"Tokenizer saved to standard directory: {tokenizer_dir}")

    # --- token_bytes.ptの作成 (損失評価に必要) ---
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)
    token_bytes_tensor = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes_tensor, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # --- レポートへのログ記録 ---
    from nanochat.report import get_report
    get_report().log(section="Tokenizer training", data=[vars(args), {"train_time": train_time}])
    print("\n--- Japanese tokenizer training finished successfully! ---")

if __name__ == "__main__":
    main()

