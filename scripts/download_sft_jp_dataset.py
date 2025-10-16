import os
import requests
from urllib.parse import urlparse

# --- 設定 ---
DATASET_URLS = [
    "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00000-of-00002.arrow",
    "https://huggingface.co/datasets/Ego/jpflan/resolve/main/data-00001-of-00002.arrow"
]
DATA_DIR_SFT_JP = "jp_sft_data" # 日本語SFTデータ用の専用キャッシュディレクトリ

def get_sft_jp_data_dir():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    sft_data_dir = os.path.join(base_dir, DATA_DIR_SFT_JP)
    os.makedirs(sft_data_dir, exist_ok=True)
    return sft_data_dir

def download_sft_file(url):
    """
    指定されたURLからファイルをダウンロードし、永続的に保存する。
    """
    data_dir = get_sft_jp_data_dir()
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return

    print(f"Downloading {filename} to {data_dir}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        # エラーが発生した場合は、不完全なファイルを削除
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

if __name__ == "__main__":
    print(f"--- Downloading JpFlan dataset for SFT ---")
    for url in DATASET_URLS:
        download_sft_file(url)
    print("--- SFT dataset download complete. ---")

