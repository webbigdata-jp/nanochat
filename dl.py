import os
import requests
import urllib.parse
import pyarrow.parquet as pq

# --- 設定 ---
DATASET_REPO_ID = "kajuma/ABEJA-CC-JA-edu"
CONFIG_NAME = "10%"
SPLIT = "train"
SHARD_INDEX = 0  # 最初のファイル(0番目)だけを調査します
TOTAL_SHARDS = 378
DOWNLOAD_CACHE_DIR = "download_cache"

def check_parquet_schema(repo_id, config_name, split, shard_index, total_shards, cache_dir):
    """
    Parquetファイルを1つだけダウンロードし、そのスキーマ（カラム構造）を表示する。
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    encoded_config = urllib.parse.quote(config_name)
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    filename = f"{split}-{shard_index:05d}-of-{total_shards:05d}.parquet"
    url = f"{base_url}/{encoded_config}/{filename}"
    local_path = os.path.join(cache_dir, filename)
    
    print(f"Downloading a single file to inspect its schema:\n  {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        print("\nInspecting Parquet file schema...")
        pf = pq.ParquetFile(local_path)
        
        # スキーマ情報を表示
        print("--- Schema Information ---")
        print(pf.schema)
        print("--------------------------")
        
        # 最初の数行のサンプルも表示してみる
        print("\n--- Data Sample (first 5 rows) ---")
        reader = pf.reader()
        first_batch = reader.read_next_batch()
        print(first_batch.to_pandas().head())
        print("----------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 調査が終わったらファイルを削除
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"\nCleaned up temporary file: {local_path}")

if __name__ == "__main__":
    check_parquet_schema(
        DATASET_REPO_ID, CONFIG_NAME, SPLIT, SHARD_INDEX, TOTAL_SHARDS, DOWNLOAD_CACHE_DIR
    )

