"""
Download VoxForge dataset to your "to_folder".
This code uses a file list from TFDS, downloads .tgz files, and extract them.
The definition of labels and data splits is available in evar/metadata/voxforge.csv.

Following TFDS implementation for the details.

## Usage

'''sh
python download_voxforge.py <to_folder>
'''

## Reference

- [1] http://www.voxforge.org/
- [2] TFDS: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/voxforge.py

@article{maclean2018voxforge,
    title={Voxforge},
    author={MacLean, Ken},
    journal={Ken MacLean.[Online]. Available: http://www.voxforge.org/home.[Acedido em 2012]},
    year={2018}
}
"""

import urllib.request
import shutil
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import fire
import socks
import socket
import requests
import time


TFDS_URL = 'https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt'


def _download_extract_worker(args):
    url, filename, dest_path, max_retries = args

    if (Path(dest_path)/Path(filename).stem).exists():
        #print(' skip', Path(filename).stem)
        #print('.', end='')
        return filename

    tmpfile = '/tmp/' + filename
    
    # 使用 requests 下载文件，带重试机制
    proxies = {
        "http": "socks5h://127.0.0.1:1080",
        "https": "socks5h://127.0.0.1:1080",
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get('http://' + url, proxies=proxies, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(tmpfile, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            break  # 下载成功，跳出重试循环
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f'ERROR to download {url} after {max_retries} attempts: {e}')
                return None
            else:
                print(f'Retry {attempt + 1}/{max_retries} for {url}: {e}')
                time.sleep(2 ** attempt)  # 指数退避：2, 4, 8, 16... 秒
        
    # 解压文件，也带重试机制
    for attempt in range(max_retries):
        try:
            shutil.unpack_archive(tmpfile, dest_path)
            break  # 解压成功，跳出重试循环
        except Exception as e:
            if attempt == max_retries - 1:
                print(f'ERROR to extract {url} after {max_retries} attempts: {e}')
                return None
            else:
                print(f'Retry {attempt + 1}/{max_retries} for extracting {url}: {e}')
                time.sleep(1)  # 解压重试等待时间较短

    os.remove(tmpfile)
    return filename


def download_extract_voxforge(dest_path, max_retries=3, max_workers=4):
    """
    下载并解压 VoxForge 数据集
    
    Args:
        dest_path: 目标路径
        max_retries: 最大重试次数，默认3次
        max_workers: 最大并发数，默认4个进程
    """
    # 设置代理
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    socket.socket = socks.socksocket
    
    # 使用 requests 获取 URL 列表，也带重试机制
    proxies = {
        "http": "socks5h://127.0.0.1:1080",
        "https": "socks5h://127.0.0.1:1080",
    }
    time.sleep(1)
    for attempt in range(max_retries):
        try:
            response = requests.get(TFDS_URL, proxies=proxies, timeout=30)
            response.raise_for_status()
            lines = response.text.splitlines()
            
            urls = [line.strip() for line in lines]
            filenames = [url.split('/')[-1] for url in urls]
            assert len(set(filenames)) == len(urls)
            break  # 成功获取URL列表，跳出重试循环
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"ERROR: Failed to fetch URL list after {max_retries} attempts: {e}")
                return
            else:
                print(f"Retry {attempt + 1}/{max_retries} for fetching URL list: {e}")
                time.sleep(2 ** attempt)

    print(f'Downloading voxforge for {len(urls)} tgz archives with max {max_retries} retries per file.')
    print(f'Using {max_workers} concurrent workers.')
    Path(dest_path).mkdir(exist_ok=True, parents=True)
    with Pool(max_workers) as p:
        args = [[url, filename, dest_path, max_retries] for url, filename in zip(urls, filenames)]
        results = list(tqdm(p.imap(_download_extract_worker, args), total=len(args)))

    # 统计成功和失败的文件
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    
    print(f'finished. Successfully processed: {len(successful)}, Failed: {failed}')
    
    if failed > 0:
        print(f'WARNING: {failed} files failed to process after {max_retries} retries each. Check the error messages above.')


if __name__ == "__main__":
    fire.Fire(download_extract_voxforge)
