# scripts/download_jp_dataset.py

import os
import argparse
import time
import requests
import urllib.parse
from multiprocessing import Pool

# --- 設定 ---
REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
MAX_SHARD = 377 # 0から377まで
DATA_DIR_JP = "jp_base_data" # 日本語データ用の専用キャッシュディレクトリ

def get_jp_data_dir():
    # nanochatの標準ベースディレクトリを取得
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    jp_data_dir = os.path.join(base_dir, DATA_DIR_JP)
    os.makedirs(jp_data_dir, exist_ok=True)
    return jp_data_dir

def download_single_shard(shard_index):
    """
    指定されたシャードをダウンロードし、永続的に保存する。
    """
    data_dir = get_jp_data_dir()
    filename = f"{SPLIT}-{shard_index:05d}-of-{MAX_SHARD+1:05d}.parquet"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True

    encoded_config = urllib.parse.quote(CONFIG_NAME)
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{encoded_config}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts.")
                return False
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Japanese dataset shards")
    parser.add_argument("-n", "--num-shards", type=int, default=MAX_SHARD + 1, help="Number of shards to download")
    parser.add_argument("-w", "--num-workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    num_to_download = min(args.num_shards, MAX_SHARD + 1)
    ids_to_download = list(range(num_to_download))
    print(f"Downloading {len(ids_to_download)} shards to '{get_jp_data_dir()}' using {args.num_workers} workers...")
    
    with Pool(processes=args.num_workers) as pool:
        pool.map(download_single_shard, ids_to_download)
    
    print("Download complete.")

