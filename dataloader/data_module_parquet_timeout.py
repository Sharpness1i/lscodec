import torch
import soundfile as sf
from typing import Union, List
from pathlib import Path
import numpy as np
import random
import pytorch_lightning as pl
import torch.utils
import torch.utils.data
from copy import deepcopy
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import queue
import torch.nn.functional as F

import threading
import librosa
import yaml
import time
import pyarrow.parquet as pq
from io import BytesIO
import torchaudio
import warnings
warnings.filterwarnings("ignore")


class TrainDataLoadIter:
    def __init__(
        self,
        speech_scp_path: Union[str, Path, List],
        batch_size: int = 1,
        cut_duration: Union[float, List[float]] = 5.0,
        num_workers: int = 1,
        prefetch: int = 0,
        save_ckpt_step: int = 16000,
        samples_per_epoch: int = 10000,
    ):  
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.is_train = True
        self.batch_size = batch_size
        self.cut_duration = cut_duration
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.samples_per_epoch = samples_per_epoch
        self.timeout = 2.0 # seconds
        self.current_shard = None
        self.shard_idx = 0
        self.speech_list = self.load_scp_to_list(speech_scp_path)
        self.speech_list = [str(x) for x in self.speech_list]
        
    
        self.shards = self.speech_list[self.rank::self.world_size] or [self.speech_list[self.rank % len(self.speech_list)]]
        
        if not self.shards:  # 防越界
            self.shards = [self.speech_list[self.rank % len(self.speech_list)]]
            
        # parquet shard state
        self.current_shard_data = None
        self.current_shard_len = 0
        self.shard_idx = 0
        self.row_idx = 0
        
        # open first shard
        self._load_parquet_shard(self.shards[self.shard_idx])
        
        # resample 24k → 16k
        self.resampler = torchaudio.transforms.Resample(24000, 16000)
        
    
    def load_scp_to_list(self, scp_path):
        path_list = []
        if not isinstance(scp_path, List):
            scp_path = [scp_path]
        for p in scp_path:
            with open(p, 'r') as f:
                for line in f:
                    path = line.strip().split(' ')[-1]
                    path_list.append(path)
        return path_list
    
    def _load_parquet_shard(self, path):
        table = pq.read_table(path)
        self.current_shard_data = table.to_pandas()
        self.current_shard_len = len(self.current_shard_data)
        self.row_idx = 0
        self.shard_idx = (self.shard_idx + 1) % len(self.shards)
    
    def pad_or_cut_wav(self, wav, length, offset=None):
        # wav: [1, T]
        if wav.shape[-1] < length: # pad
            wav = np.pad(wav, [(0, 0), (0, length - wav.shape[-1])], mode='wrap')
            return wav, None
        else: # cut
            if offset is None:
                offset = random.randint(0, wav.shape[-1] - length)
            wav = wav[..., offset: offset + length]
            return wav, offset

    def normalize_wav(self, wav, low=0.1, high=0.99):
        max_value = np.max(np.abs(wav)) + 1e-5
        target_value = random.uniform(low, high)
        wav = wav * target_value / max_value
        return wav
    
     # ========== timeout audio decode ==========
    def _load_audio_queue(self, audio_bytes, q):
        try:
            wav, _ = torchaudio.load(BytesIO(audio_bytes))
            q.put(wav)
        except Exception as e:
            q.put(e)
    
    def _load_audio_with_timeout(self, audio_bytes):
        q = queue.Queue()
        th = threading.Thread(target=self._load_audio_queue, args=(audio_bytes, q))
        th.start()
        th.join(self.timeout)
        if th.is_alive():
            return None
        out = q.get()
        return None if isinstance(out, Exception) else out
    
    
    def get_next_utt(self):
        while True:
            if self.current_shard_data is None or self.row_idx >= self.current_shard_len:
                self._load_parquet_shard(self.shards[self.shard_idx])

            row = self.current_shard_data.iloc[self.row_idx]
            self.row_idx += 1

            audio_bytes = row["audio_data"]
            sr = row.get("sr", 24000)  # default 24k if missing

            wav = self._load_audio_with_timeout(audio_bytes)
            if wav is None:
                continue

            # mono
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)

            # pad/cut 5s
            target = int(self.cut_duration * sr)
            T = wav.size(1)
            if T < target:
                wav = F.pad(wav, (0, target - T))
            else:
                start = random.randint(0, T - target)
                wav = wav[:, start:start + target]

            # normalize
            wav = wav * (0.9 / (wav.abs().max() + 1e-7))

            wav_16k = self.resampler(wav)
            return wav, wav_16k
       
    

    def process_one_sample(self):
        w24, w16 = self.get_next_utt()
        return w24, w16

    def data_iter_fn(self, q, event):
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for _ in range(len(self)):
            futures = [executor.submit(self.process_one_sample) for _ in range(self.batch_size)]
            wav24s, wav16s = [], []
            for f in futures:
                w24, w16 = f.result()
                wav24s.append(w24)
                wav16s.append(w16)
            q.put((torch.cat(wav24s, 0), torch.cat(wav16s, 0)))
        event.set()

    
    def __iter__(self):
        q = queue.Queue(self.prefetch + 1)
        ev = threading.Event()
        threading.Thread(target=self.data_iter_fn, args=(q, ev), daemon=True).start()

        while not ev.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        return self.samples_per_epoch // (self.world_size * self.batch_size)

class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_kwargs,
    ):
        super().__init__()
        self.train_kwargs = train_kwargs


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = TrainDataLoadIter(**self.train_kwargs)
       

    def train_dataloader(self):
        return self.train_iter



