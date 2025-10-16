import os
import time
import requests
import urllib.parse
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer

# --- 設定項目 ---
DATASET_REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
TOTAL_SHARDS = 378

TARGET_CHARS = 2_000_000_000
TARGET_CHARS =   800_000_000 # $<-ng
TARGET_CHARS =   500_000_000 
VOCAB_SIZE = 65536
OUTPUT_DIR = "japanese_tokenizer"
DOWNLOAD_CACHE_DIR = "download_cache"

def direct_download_iterator(repo_id, config_name, split, total_shards, cache_dir):
    """
    ファイルI/Oに専念するイテレータ。テキストのバッチ（リスト）をyieldする。
    文字数制限は行わない。
    """
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
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}"); continue

        print(f"Processing {os.path.basename(local_path)}...")
        try:
            pf = pq.ParquetFile(local_path)
            for batch in pf.iter_batches(batch_size=8192):
                yield batch.to_pydict()['content']
        finally:
            if os.path.exists(local_path): os.remove(local_path)

def limited_char_iterator(base_iterator, target_chars):
    """
    ベースイテレータをラップし、文字列を1行ずつyieldしながら文字数をカウントし、
    上限に達したら完全に停止する。
    """
    total_chars = 0
    start_time = time.time()
    for text_batch in base_iterator:
        for text in text_batch:
            if text is None: continue
            
            text_len = len(text)
            if total_chars + text_len >= target_chars:
                remaining_chars = target_chars - total_chars
                yield text[:remaining_chars]
                total_chars += remaining_chars
                print(f"\nTarget characters reached. Total processed characters: {total_chars:,}")
                return # ★★★ ここでジェネレータを完全に停止させる
            else:
                yield text
                total_chars += text_len
        
        elapsed_time = time.time() - start_time
        chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0
        print(f"\rProcessed {total_chars:,} / {target_chars:,} characters ({total_chars/target_chars:.2%}) | {chars_per_sec:,.0f} chars/sec", end="")

# --- メインの実行部分 ---
if __name__ == "__main__":
    print("--- Starting Japanese Tokenizer Training Test (Final Architecture) ---")
    
    # 1. ベースとなるイテレータを作成
    base_iterator = direct_download_iterator(
        DATASET_REPO_ID, CONFIG_NAME, SPLIT, TOTAL_SHARDS, DOWNLOAD_CACHE_DIR
    )
    
    # 2. 文字数制限を行うラッパーイテレータを作成
    final_iterator = limited_char_iterator(base_iterator, TARGET_CHARS)

    # 3. rustbpeに最終的なイテレータを渡す
    print(f"\nStarting tokenizer training with vocab size: {VOCAB_SIZE}...")
    tokenizer = RustBPETokenizer.train_from_iterator(final_iterator, vocab_size=VOCAB_SIZE)
    
    print("\nTokenizer training completed successfully!")

    print(f"Saving tokenizer to '{OUTPUT_DIR}'...")
    tokenizer.save(OUTPUT_DIR)
    
    print("\n--- Testing the trained tokenizer ---")
    # (テスト部分は変更なし)
    test_text = "これは日本語のトークナイザーの性能をテストするための例文です。"
    encoded_ids = tokenizer.encode(test_text)
    print(f"Original Text: {test_text}")
    print(f"Encoded IDs: {encoded_ids}")
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded Text: {decoded_text}")
    tokens = [tokenizer.decode([id]) for id in encoded_ids]
    print(f"Tokens: {'|'.join(tokens)}")
    
    print("\n--- Test Finished ---")

