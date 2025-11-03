import torch
import logging
import random
import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw
import os
AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def parquet_opener(data, mode='train', tts_data={}):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':

                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('Failed to open {}, ex info {}'.format(url, ex))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=1024,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):

    for sample in data:
        utt_id = sample.get('utt', '[unknown_utt]')
        utt_embedding = sample.get('utt_embedding', None)
        if utt_embedding is None:
            print(utt_id, "缺失 utt_embedding")
            continue
        audio_data = sample.get('audio_data', None)
        if audio_data is None:
            print(utt_id, "缺失 audio_data")
            continue
        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(audio_data))
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)
        del sample['audio_data']

        
        
        sample_rate = sample.get('sample_rate', None)
        if sample['speech'] is None or sample_rate is None:
            print(utt_id, "缺失 speech 或 sample_rate")
            continue
        num_frames = sample['speech'].size(1) / sample_rate * 100

        if num_frames > max_length:
            print(utt_id, f"帧数太长: {num_frames:.2f} > {max_length}")
            continue

        yield sample


def resample(data, resample_rate=16000, min_sample_rate=16000, mode='train'):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            sample['speech_16k'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
            
        sample['speech'] = waveform
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  mode='train'):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        #assert 'text_token' in sample
        waveform = sample['speech']
        mat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)  # mel 
        sample['speech_feat'] = mat
        yield sample


def compute_f0(data, sample_rate, hop_size, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    frame_period = hop_size * 1000 / sample_rate
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        _f0, t = pw.harvest(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period)
        if sum(_f0 != 0) < 5: # this happens when the algorithm fails
            _f0, t = pw.dio(waveform.squeeze(dim=0).numpy().astype('double'), sample_rate, frame_period=frame_period) # if harvest fails, try dio
        f0 = pw.stonemask(waveform.squeeze(dim=0).numpy().astype('double'), _f0, t, sample_rate)
        f0 = F.interpolate(torch.from_numpy(f0).view(1, 1, -1), size=sample['speech_feat'].shape[0], mode='linear').view(-1)
        sample['pitch_feat'] = f0
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ Parse utt_embedding/spk_embedding

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample



def shuffle(data, shuffle_size=10000, mode='train'):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0) # 为什么这里取得是speech_feat ? 
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ Wrapper for static/dynamic batch
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('Unsupported batch type {}'.format(batch_type))

def get_position_ids(lengths):
    max_len = lengths.max().item()
    arange_ids = torch.arange(max_len).unsqueeze(0).expand(lengths.size(0), -1)  
    mask = arange_ids < lengths.unsqueeze(1)  
    position_ids = arange_ids * mask
    return position_ids

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:

    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def right_pad_sequence(sequences, batch_first=False, padding_value=0):
    """类似 torch.nn.utils.rnn.pad_sequence，但做右 padding"""
    max_len = max([seq.size(0) for seq in sequences])
    out_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if batch_first:
            out_tensor[i, :length] = seq  # 把真实 token 放左边
        else:
            out_tensor[:length, i] = seq
    return out_tensor

def left_pad_sequence(sequences, batch_first=False, padding_value=0):
    """类似 torch.nn.utils.rnn.pad_sequence，但做左 padding"""
    max_len = max([seq.size(0) for seq in sequences])
    out_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if batch_first:
            out_tensor[i, -length:] = seq  # 把真实 token 放右边
        else:
            out_tensor[-length:, i] = seq
    return out_tensor


def padding(data, use_spk_embedding, mode='train', gan=False):
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample], dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)
        speaker_ids =  [sample[i]['spk'] for i in order]
        waveform_list = [sample[i]['speech'].squeeze(0).clone() for i in order]
        waveform_list_16k = [sample[i]['speech_16k'].squeeze(0).clone() for i in order]
        
        waveform_lens = [x.size(0) for x in waveform_list]   
        waveform_lens_16k =  [x.size(0) for x in waveform_list_16k]   
        # pad waveform
        waveforms = pad_sequence(waveform_list,batch_first=True,padding_value=0.0)
        waveforms_16k = pad_sequence(waveform_list_16k,batch_first=True,padding_value=0.0)
        
        utts = [sample[i]['utt'] for i in order]
        
        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)

        utt_embedding = torch.stack([sample[i]['utt_embedding'].clone() for i in order], dim=0)

        batch = {
            "utts": utts,
            "tgt_speech_feat": speech_feat,
            "tgt_speech_feat_len": speech_feat_len,
            "tgt_speaker_embedding": utt_embedding,
            "task": "TTS",
            "speech":waveforms,
            "speech_lens": waveform_lens,
            "speech_16k_lens": waveform_lens_16k,
            "speakers_id" :speaker_ids,
            "speech_16k":waveforms_16k,
        }
        if gan is True: # true
            # in gan train, we need pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len

        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
   
        yield batch
