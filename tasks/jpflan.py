# tasks/jpflan.py

import os
import sys
import pyarrow.ipc
from typing import List, Dict, Any

# _cached_data変数は不要になったので削除

class JpFlan:
    """
    ローカルにキャッシュされた Ego/jpflan データセットをロードし、
    nanochatの会話形式で提供するタスククラス。
    """
    EXPECTED_FILES = [
        "data-00000-of-00002.arrow",
        "data-00001-of-00002.arrow"
    ]
    
    def __init__(self, split="train"):
        # split引数は互換性のために維持
        self.data = self._load_from_cache()
        if split == "train":
             print(f"Total {len(self.data):,} records loaded for JpFlan training from local cache.")

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

    def _load_from_cache(self) -> List[Dict[str, Any]]:
        """
        ローカルのキャッシュディレクトリからArrowファイルを読み込む。
        """
        # download_sft_jp_dataset.py と同じパス解決ロジックを使用
        from scripts.download_sft_jp_dataset import get_sft_jp_data_dir
        data_dir = get_sft_jp_data_dir()
        
        all_data = []
        for filename in self.EXPECTED_FILES:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"FATAL: SFT data file not found at {filepath}", file=sys.stderr)
                print("Please run 'python -m scripts.download_sft_jp_dataset' first.", file=sys.stderr)
                sys.exit(1)
            
            print(f"Loading SFT data from {filepath}...")
            with open(filepath, 'rb') as f:
                with pyarrow.ipc.open_stream(f) as reader:
                    for batch in reader:
                        all_data.extend(batch.to_pylist())
        
        return all_data

