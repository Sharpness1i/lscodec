import os
import torch
from tqdm import tqdm
from transformers import AutoModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.dataset.dataset import Dataset

import os
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() in ("1", "true", "yes")
if DEBUG_MODE:
    import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();

import torchaudio
from io import BytesIO

TARGET_SR = 16000
MAX_SEC = 12
MAX_SAMPLES = TARGET_SR * MAX_SEC

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchaudio
import pyarrow.parquet as pq
from io import BytesIO
from transformers import AutoModel
from tqdm import tqdm

###############################################
# CONFIG
###############################################

DATA_LIST = "/primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list"
TARGET_SR = 16000
MAX_SEC = 15                                # 15秒截断（你可以改）
MAX_SAMPLES = TARGET_SR * MAX_SEC
OUT_DIR = "/primus_biz_workspace/zhangboyang.zby/data/emilia/train/teacher_feats"
os.makedirs(OUT_DIR, exist_ok=True)

WAVLM_DIR = os.environ.get("WAVLM_DIR")
assert WAVLM_DIR is not None, "请 export WAVLM_DIR=/path/to/wavlm_large"

###############################################
# LOAD TEACHER
###############################################

device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = AutoModel.from_pretrained(WAVLM_DIR).to(device).eval()
print("✅ Loaded WavLM teacher")

###############################################
# DATA STREAM (your parquet_opener style)
###############################################

def parquet_stream(data_list):
    with open(data_list) as f:
        for line in f:
            url = line.strip()
            if not url:
                continue

            try:
               
                pf = pq.ParquetFile(url)
                for batch in pf.iter_batches(batch_size=64):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        # yield raw parquet row as python dict
                        yield {
                            "src": url,
                            "utt": row["utt"],
                            "audio_data": row["audio_data"]
                        }
            except Exception as e:
                print(f"❌ FAIL reading {url}, err={e}")
                continue

###############################################
# AUDIO DECODE + RESAMPLE + CROP
###############################################

def decode_16k_clean(data_stream):
    for x in data_stream:
        wav, sr = torchaudio.load(BytesIO(x["audio_data"]))
        wav = wav.mean(0, keepdim=True)  # mono

        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

        wav = wav.squeeze(0)

        if wav.shape[0] > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]

        x["wav_16k"] = wav
        x["wav_len"] = wav.shape[0]

        yield x

###############################################
# TEACHER EMBEDDING + SAVE
###############################################

@torch.inference_mode()
def dump_teacher():
    stream = parquet_stream(DATA_LIST)
    stream = decode_16k_clean(stream)

    for item in tqdm(stream, desc="Extracting WavLM feats"):
        utt = item["utt"]
        wav = item["wav_16k"]

        # shape [T] -> [1, T]
        feat = teacher(wav.unsqueeze(0).to(device)).last_hidden_state.cpu()

        save_path = os.path.join(OUT_DIR, f"{utt}.pt")
        torch.save(
            {
                "utt": utt,
                "feat": feat,         # [T', C]
                "wav_len": item["wav_len"],  # ORIGINAL waveform length
                "src": item["src"]
            },
            save_path
        )

        # print(f"✅ saved {utt}")

###############################################
# RUN
###############################################

if __name__ == "__main__":
    dump_teacher()
    print(f"\n🎉 DONE! Saved to {OUT_DIR}\n")