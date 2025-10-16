# nanochat/dataset_jp.py

import os
import time
import requests
import urllib.parse
import pyarrow.parquet as pq


def direct_download_iterator(repo_id, config_name, split, total_shards, cache_dir, start=0, step=1):
    """
    ファイルI/Oに専念するイテレータ。テキストのバッチ（リスト）をyieldする。
    """
    os.makedirs(cache_dir, exist_ok=True)
    encoded_config = urllib.parse.quote(config_name)
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    
    for i in range(start, total_shards, step):
        filename = f"{split}-{i:05d}-of-{total_shards:05d}.parquet"
        url = f"{base_url}/{encoded_config}/{filename}"
        local_path = os.path.join(cache_dir, filename)
        
        print(f"\nDownloading: {url}")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 404:
                print(f"File not found, stopping download loop: {url}")
                break
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            continue

        print(f"Processing {os.path.basename(local_path)}...")
        try:
            pf = pq.ParquetFile(local_path)
            for batch in pf.iter_batches(batch_size=8192):
                yield batch.to_pydict()['content']
        finally:
            # ダウンロードした一時ファイルは処理後に削除
            if os.path.exists(local_path):
                os.remove(local_path)

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
                return # ジェネレータを完全に停止
            else:
                yield text
                total_chars += text_len
        
        elapsed_time = time.time() - start_time
        chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0
        print(f"\rProcessed {total_chars:,} / {target_chars:,} characters ({total_chars/target_chars:.2%}) | {chars_per_sec:,.0f} chars/sec", end="")


