# tasks/jpflan.py

import sys
import requests
import pyarrow.ipc
from typing import List, Dict, Any

# データは一度ロードしたらメモリにキャッシュしておくための変数
_cached_data = None

def download_and_load_data(urls: List[str]) -> List[Dict[str, Any]]:
    """
    複数のURLからArrowファイルを直接ダウンロードし、内容をリストとして結合する。
    """
    global _cached_data
    if _cached_data is not None:
        print("Using cached JpFlan data.")
        return _cached_data

    all_data = []
    for i, url in enumerate(urls):
        print(f"Downloading JpFlan data from {url} ({i+1}/{len(urls)})...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with pyarrow.ipc.open_stream(response.content) as reader:
                for batch in reader:
                    all_data.extend(batch.to_pylist())
            print(f"Successfully loaded {len(all_data)} records so far.")
        except Exception as e:
            print(f"Error processing Arrow file from {url}: {e}", file=sys.stderr)
            sys.exit(1)
            
    _cached_data = all_data
    return all_data

class JpFlan:
    """
    Ego/jpflan データセットをロードし、nanochatの会話形式で提供するタスククラス。
    """
    DATASET_URLS = [
        "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00000-of-00002.arrow",
        "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00001-of-00002.arrow"
    ]
    
    def __init__(self, split="train"):
        # split引数は他のタスクとの互換性のためにあるが、このデータセットでは使用しない
        self.data = download_and_load_data(self.DATASET_URLS)
        if split == "train":
             print(f"Total {len(self.data):,} records loaded for JpFlan training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        
        # [INST] タグを除去してユーザーの入力とする
        user_content = item['input'].replace("[INST]", "").replace("[/INST]", "").strip()
        assistant_content = item['output']
        
        # nanochatが要求する会話形式に変換
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }


