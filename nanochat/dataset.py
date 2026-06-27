"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import random
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import resource
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
index_to_filename = lambda index: f"shard_{index:05d}.parquet"

OPENWEBMATH_URL = "https://huggingface.co/datasets/open-web-math/open-web-math/resolve/refs%2Fconvert%2Fparquet/default/train"
OPENWEBMATH_MAX_SHARD = 113
openwebmath_index_to_filename = lambda index: f"{index:04d}.parquet"

# 你提供的直接下载链接
GSM8K_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet?download=true"
AQUA_RAT_URL = "https://huggingface.co/datasets/deepmind/aqua_rat/resolve/main/raw/train-00000-of-00001.parquet?download=true"

from nanochat.common import get_base_dir
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")
MID_TRAIN_DATA_DIR = os.path.join(base_dir, "mid_train_data")
TEMP_DOWNLOAD_DIR = os.path.join(base_dir, "tmp_mid_download")
MATH_DIR = os.path.join(base_dir, "math_datasets")
os.makedirs(MATH_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# System & HTTP session
# -----------------------------------------------------------------------------
def set_system_limits():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(4096, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
set_system_limits()

def init_global_session():
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.2, status_forcelist=[429,500,502,503,504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip"})
    session.verify = False
    session.timeout = 30
    return session
GLOBAL_SESSION = init_global_session()

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    if data_dir is None:
        data_dir = os.path.join(base_dir, "base_data_climbmix")
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")])
    return [os.path.join(data_dir, f) for f in files]

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
def download_single_file(index, data_dir, dataset_type):
    if dataset_type == "climb":
        fname = index_to_filename(index)
        url = f"{BASE_URL}/{fname}"
    elif dataset_type == "openwebmath":
        fname = openwebmath_index_to_filename(index)
        url = f"{OPENWEBMATH_URL}/{fname}"
    elif dataset_type == "gsm8k":
        fname = "gsm8k_train.parquet"
        url = GSM8K_URL
    elif dataset_type == "aqua_rat":
        fname = "aqua_rat_train.parquet"
        url = AQUA_RAT_URL
    else:
        return False

    fpath = os.path.join(data_dir, fname)
    tmp_path = fpath + ".tmp"

    if os.path.exists(fpath):
        return True

    try:
        resume_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={resume_size}-"} if resume_size > 0 else {}

        resp = GLOBAL_SESSION.get(url, stream=True, headers=headers, timeout=120)
        resp.raise_for_status()

        with open(tmp_path, 'ab', buffering=1024*1024) as f:
            for chunk in resp.iter_content(chunk_size=16*1024*1024):
                if chunk:
                    f.write(chunk)

        os.rename(tmp_path, fpath)
        return True
    except Exception as e:
        return False

# -----------------------------------------------------------------------------
# 格式转换：把 question + answer 拼成 text
# -----------------------------------------------------------------------------
def convert_gsm8k_to_text(input_path, output_path):
    if os.path.exists(output_path):
        return
    df = pd.read_parquet(input_path)
    df["text"] = "Question: " + df["question"].str.strip() + "\nAnswer: " + df["answer"].str.strip()
    df[["text"]].to_parquet(output_path, row_group_size=1024)

def convert_aqua_rat_to_text(input_path, output_path):
    if os.path.exists(output_path):
        return

    df = pd.read_parquet(input_path)

    texts = []
    for _, row in df.iterrows():
        q = row["question"].strip()
        opts = row["options"]
        rationale = row["rationale"].strip()
        correct = row["correct"].strip()

        opt_str = "\n".join(opts)

        text = (
            f"Question: {q}\n"
            f"Options:\n{opt_str}\n"
            f"Rationale: {rationale}\n"
            f"Answer: {correct}"
        )
        texts.append(text)

    pd.DataFrame({"text": texts}).to_parquet(output_path, row_group_size=1024)

# -----------------------------------------------------------------------------
# Streaming 读取
# -----------------------------------------------------------------------------
def stream_texts_uniform(file_list, shuffle_buffer=10000):
    import random
    random.seed(42)

    readers = []
    for f in file_list:
        try:
            readers.append(pq.ParquetFile(f))
        except:
            continue

    if not readers:
        return

    row_group_indices = [0] * len(readers)
    current_batches = [None] * len(readers)
    current_ptrs = [0] * len(readers)

    while True:
        try:
            active = []
            for i, r in enumerate(readers):
                if row_group_indices[i] < r.num_row_groups or (current_batches[i] is not None and current_ptrs[i] < len(current_batches[i])):
                    active.append(i)

            if not active:
                return

            idx = random.choice(active)
            reader = readers[idx]

            if current_batches[idx] is None or current_ptrs[idx] >= len(current_batches[idx]):
                try:
                    rg = row_group_indices[idx]
                    batch = reader.read_row_group(rg, columns=["text"])
                    current_batches[idx] = batch["text"].to_pylist()
                    current_ptrs[idx] = 0
                    row_group_indices[idx] += 1
                except:
                    row_group_indices[idx] += 1
                    continue

            text = current_batches[idx][current_ptrs[idx]]
            current_ptrs[idx] += 1
            if text and len(text.strip()) > 0:
                yield text
        except GeneratorExit:
            return
        except:
            continue

# -----------------------------------------------------------------------------
# 混合输出
# -----------------------------------------------------------------------------
def stream_mix(climb_files, math_files, out_dir, num_output_files):
    os.makedirs(out_dir, exist_ok=True)

    # 生成器 - 改成循环生成器，不会耗尽
    def endless_generator(gen_func, files):
        while True:
            gen = gen_func(files)
            yield from gen

    climb_gen = endless_generator(stream_texts_uniform, climb_files)
    math_gen = endless_generator(stream_texts_uniform, math_files)

    BATCH_PER_FILE = 10000
    current = []
    file_idx = 0
    total = num_output_files * BATCH_PER_FILE
    pbar = tqdm(desc="Mixing data", total=total)

    try:
        while file_idx < num_output_files:
            if random.random() < 0.7:
                txt = next(climb_gen)
            else:
                txt = next(math_gen)

            current.append(txt)
            pbar.update(1)

            # 写入文件
            if len(current) >= BATCH_PER_FILE:
                out_path = os.path.join(out_dir, f"mixed_{file_idx:04d}.parquet")
                pq.write_table(pa.table({"text": current}), out_path, row_group_size=1024)
                current = []
                file_idx += 1

    finally:
        # 剩余数据写入
        if current and file_idx < num_output_files:
            out_path = os.path.join(out_dir, f"mixed_{file_idx:04d}.parquet")
            pq.write_table(pa.table({"text": current}), out_path)

        del climb_gen
        del math_gen
        pbar.close()

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & mix dataset")
    parser.add_argument("-n", "--num-files", type=int, required=True)
    parser.add_argument("-w", "--num-workers", type=int, default=16)
    parser.add_argument("-d", "--dataset", choices=["base", "mid_train"], default="mid_train")
    args = parser.parse_args()

    if args.dataset == "base":
        os.makedirs(DATA_DIR, exist_ok=True)
        n = min(args.num_files, MAX_SHARD)
        ids = list(range(n))
        print(f"Downloading ClimbMix {len(ids)} files")
        with ThreadPool(args.num_workers) as pool:
            pool.map(lambda i: download_single_file(i, DATA_DIR, "climb"), ids)

    elif args.dataset == "mid_train":
        os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(MATH_DIR, exist_ok=True)
        os.makedirs(MID_TRAIN_DATA_DIR, exist_ok=True)

        # 下载 ClimbMix
        num_climb = 20
        climb_start = max(0, MAX_SHARD - num_climb + 1)
        climb_ids = list(range(climb_start, MAX_SHARD + 1))
        # print(f"⬇️ 下载 ClimbMix: {climb_ids}")

        with ThreadPool(args.num_workers) as pool:
            pool.map(lambda i: download_single_file(i, TEMP_DOWNLOAD_DIR, "climb"), climb_ids)

        climb_files = [os.path.join(TEMP_DOWNLOAD_DIR, index_to_filename(i)) for i in climb_ids]
        climb_files = [f for f in climb_files if os.path.exists(f)]

        # 下载 GSM8K + AQUA-RAT
        gsm_input = os.path.join(MATH_DIR, "gsm8k_train.parquet")
        aqua_input = os.path.join(MATH_DIR, "aqua_rat_train.parquet")
        download_single_file(0, MATH_DIR, "gsm8k")
        download_single_file(0, MATH_DIR, "aqua_rat")

        # 转换为 text 列
        gsm_out = os.path.join(MATH_DIR, "gsm8k_text.parquet")
        aqua_out = os.path.join(MATH_DIR, "aqua_text.parquet")
        convert_gsm8k_to_text(gsm_input, gsm_out)
        convert_aqua_rat_to_text(aqua_input, aqua_out)

        math_files = [gsm_out, aqua_out]

        # 开始混合
        stream_mix(climb_files, math_files, MID_TRAIN_DATA_DIR, args.num_files)

        print(f"\n🎉 全部完成！输出路径: {MID_TRAIN_DATA_DIR}")