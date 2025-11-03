import queue
import random
import time
import threading
from typing import Dict, Optional, Callable, List, Generator
import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils.generic import ModelOutput
from transformers.models.qwen2 import Qwen2ForCausalLM
import torch
import torch.nn as nn
import logging
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
# from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.mask import make_pad_mask
IGNORE_ID = -100
from dataclasses import dataclass
from transformers.loss.loss_utils import ForCausalLMLoss
@dataclass
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    thinker_reply_part (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Hidden states from the thinker model that are used as input for the talker model. These represent the encoded
        response that the talker model will use to generate speech tokens.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    thinker_reply_part: torch.FloatTensor = None
    labels: Optional[torch.LongTensor] = None  #  新增一个

class TransformerLM(torch.nn.Module):
    def __init__(
            self,
            text_encoder_input_size: int,
            llm_input_size: int,
            llm_output_size: int,
            text_token_size: int,
            speech_token_size: int,
            text_encoder: torch.nn.Module,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            spk_embed_dim: int = 192,
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(),
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        # 4. sampling method
        self.sampling = sampling

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        encoder_out, encoder_mask = self.text_encoder(text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(self, sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len):
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        lm_input = [torch.concat([sos_eos_emb.squeeze(dim=0), embedding[i], text_token[i], task_id_emb.squeeze(dim=0), speech_token[i]], dim=0)
                    for i in range(len(text_token))]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text: (B, L, D)
            text_lengths: (B,)
            audio: (B, T, N) or (B, T)
            audio_lengths: (B,)
        """
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)

        # 1. prepare llm_target
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() +
                                  [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # 1. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 2. embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        # 3. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 4. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 5. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len,
                                                         task_id_emb, speech_token, speech_token_len)

        # 6. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc}

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids): # 当我们正在忽略 EOS，并且采样结果里恰好又采到了 EOS，那就得重新采，继续 while 循环。
                # 在采样过程中，即使采样结果是 EOS token，也 不把它当作停止信号。
                break
            num_trials += 1
            if num_trials > max_trials:
            
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ) -> Generator[torch.Tensor, None, None]:
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device).to(text.dtype)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(lm_input, offset=offset, required_cache_size=-1,
                                                                  att_cache=att_cache, cnn_cache=cnn_cache,
                                                                  att_mask=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]),
                                                                                                 device=lm_input.device)).to(torch.bool))
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            # force continue decode first token
            if i == 0:
                logp[:, self.speech_token_size] = -float('inf')
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            # in stream mode, yield token one by one
            # yield top_ids
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            return out_tokens

class Qwen2Encoder(torch.nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T)
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=masks,
            output_hidden_states=True,
            return_dict=True,
        )
        return outs.hidden_states[-1], masks.unsqueeze(1)

    def forward_one_step(self, xs, masks, cache=None):
        input_masks = masks[:, -1, :]
        outs = self.model(
            inputs_embeds=xs,
            attention_mask=input_masks,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
            past_key_values=cache,
        )
        xs = outs.hidden_states[-1]
        new_cache = outs.past_key_values
        return xs, new_cache



class Qwen2LM(TransformerLM):

    def __init__(
            self,
            llm_input_size: int,
            llm_output_size: int,
            speech_token_size: int,
            llm: torch.nn.Module,
            sampling: Callable,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.0,
            mix_ratio: List[int] = [5, 15],
    ):
        torch.nn.Module.__init__(self)
        #self.config = config
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        # 2. build speech token language model related modules
        
        self.thinker_to_talker_proj = nn.Linear(3584,896)
        self.reply_linear = nn.Linear(4480,896)
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 3)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 3,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 3, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(192, llm_input_size)
        # 4. sampling method
        self.sampling = sampling
        self.mix_ratio = mix_ratio

        # 5. vllm related
        self.stop_token_ids = [speech_token_size + i for i in range(3)]
        self.vllm_output_queue = {}
    

    def prepare_lm_input_target_minmo(self, prompt_hidden,prompt_len, text_token, text_token_embed, text_token_len, speech_token, speech_token_emb, speech_token_len,embedding=None):
        
        if embedding is not None:
            if embedding.shape[0] != 0:
                embedding = F.normalize(embedding, dim=1)
                embedding = self.spk_embed_affine_layer(embedding)
                embedding = embedding.unsqueeze(dim=1)
            else:
                embedding = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_hidden.dtype).to(prompt_hidden.device)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_hidden.dtype).to(prompt_hidden.device)
            
        IGNORE_ID = -100
        sos_embed = self.llm_embedding.weight[self.sos_eos].reshape(1, -1)
        task_id_embed = self.llm_embedding.weight[self.task_id].reshape(1, -1)
        text_token_embed_list = unpad_sequence(text_token_embed,text_token_len.cpu(), batch_first=True)
        prompt_hidden_list = unpad_sequence(prompt_hidden,prompt_len.cpu(), batch_first=True)
        lm_target, lm_input = [], []
   
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token_list = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
     
        speech_token_emb_list = unpad_sequence(speech_token_emb, speech_token_len.cpu(), batch_first=True)
        for i in range(len(text_token)):
            assert speech_token_list[i].size(0) == speech_token_emb_list[i].size(0)
            assert text_token_embed_list[i].size(0) == text_token[i].size(0)
            # bistream sequence
            if random.random() < 1.00 and speech_token_len[i] / text_token_len[i] > self.mix_ratio[1] / self.mix_ratio[0]:
                this_lm_target, this_lm_input = [], []
                this_lm_target.append(IGNORE_ID)
                this_lm_input.append(sos_embed)
                this_lm_target.append(IGNORE_ID)
                this_lm_input.append(embedding[i])
                
                for j in range(((text_token_len[i] + 1) / self.mix_ratio[0]).ceil().int().item()):
                    this_text_token = text_token[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]].tolist()
                    this_speech_token = speech_token_list[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]].tolist()
                    
                    if len(this_text_token) == self.mix_ratio[0]:
                        assert len(this_speech_token) == self.mix_ratio[1]
                        this_lm_target += [IGNORE_ID] * (self.mix_ratio[0] - 1)
                        this_lm_target += this_speech_token
                        this_lm_target.append(self.speech_token_size + 2) # turn id 
               
                        this_lm_input.append(text_token_embed_list[i][j * self.mix_ratio[0]: (j + 1) * self.mix_ratio[0]])
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]: (j + 1) * self.mix_ratio[1]])
                    
                    # 假设 text 与speech 是5：15，在这里耗尽了成5个为一组的text token，只有不满5个的了。
                    else: 
                        this_lm_target += [IGNORE_ID] * len(this_text_token)
                        this_lm_target += speech_token[i][j * self.mix_ratio[1]:].tolist()
                        this_lm_target.append(self.speech_token_size)

                        this_lm_input.append(text_token_embed_list[i][j * self.mix_ratio[0]:])
                        this_lm_input.append(self.llm_embedding.weight[self.task_id].reshape(1, -1))
                        this_lm_input.append(speech_token_emb[i][j * self.mix_ratio[1]:])
                        
                this_lm_target= torch.concat([torch.tensor(this_lm_target)],dim=0) 
                this_lm_input=torch.concat(this_lm_input, dim=0)
                assert this_lm_input.size(0) == this_lm_target.size(0)
            
            # unistream sequence
            else:
                this_lm_target = torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token_list[i].tolist() + [self.speech_token_size])
                this_lm_input = torch.concat([embedding[i],sos_embed, text_token_embed_list[i],
                                              task_id_embed, speech_token_emb_list[i]], dim=0)
                assert this_lm_input.size(0) == this_lm_target.size(0)
            
            if prompt_hidden_list is not None:
                IGNORE_tensor = torch.full(
                    (prompt_hidden_list[i].size(0),),
                    IGNORE_ID,
                    dtype=torch.long,
                    device=this_lm_input.device
                )
                if prompt_hidden_list[i].size(-1)!= 896:
                    prompt_hidden_list[i] = self.thinker_to_talker_proj(prompt_hidden_list[i])
                    
                this_lm_input  = torch.cat([prompt_hidden_list[i], this_lm_input], dim=0)
                this_lm_target = torch.cat([IGNORE_tensor, this_lm_target.to(this_lm_input.device)], dim=0)
            
            lm_input.append(this_lm_input)
            lm_target.append(this_lm_target)

        lm_input_len = torch.tensor([x.size(0) for x in lm_input], dtype=torch.int32)
        lm_input  = pad_sequence(lm_input,  batch_first=True, padding_value=IGNORE_ID)
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID) #  using ignore_id to pad lm_target ; 
        
        assert lm_input.size(1) == lm_target.size(1), "最终输入和目标长度不一致"
        return lm_target, lm_input, lm_input_len
    
    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        IGNORE_ID = -100        
        text_token       = batch["text_token"].to(device)
        text_token_len   = batch["text_token_len"].to(device)
        prompt_token     = batch["prompt_token"].to(device)
        prompt_token_len = batch["prompt_token_len"].to(device)
        speech_token     = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        prompt_hidden    = batch["prompt_hidden"].to(device)
        prompt_hidden_len= batch["prompt_hidden_len"].to(device)
        reply_hidden     = batch["reply_hidden"].to(device)
        reply_hidden_len = batch["reply_hidden_len"].to(device)
        reply_embed =  batch['reply_embed'].to(device)
        reply_mixed = batch['reply_mixed'].to(device)
        speaker_embed =  batch['speaker_embed'].to(device)
        speech_token_emb = self.speech_embedding(speech_token)

        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target_minmo(prompt_hidden,prompt_hidden_len, text_token, reply_mixed , text_token_len, speech_token, speech_token_emb, speech_token_len,embedding=speaker_embed)
        
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target.to(device))
        #loss = ForCausalLMLoss(logits=logits,labels=lm_target,vocab_size=self.speech_token_size + 3,ignore_index=IGNORE_ID)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 3), lm_target, ignore_label=IGNORE_ID)
        
        
        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
            thinker_reply_part=None,
        )

    def forward_dpo(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        reject_speech_token = batch['reject_speech_token'].to(device)
        reject_speech_token_len = batch['reject_speech_token_len'].to(device)

        # 1. encode text_token
        text_token_emb = self.llm.model.model.embed_tokens(text_token)

        # 2. encode speech_token
        speech_token = unpad_sequence(speech_token, speech_token_len.cpu(), batch_first=True)
        reject_speech_token = unpad_sequence(reject_speech_token, reject_speech_token_len.cpu(), batch_first=True)
        speech_token_combined = speech_token + reject_speech_token
        speech_token_combined = pad_sequence(speech_token_combined, batch_first=True, padding_value=0)
        speech_token_combined_len = torch.concat([speech_token_len, reject_speech_token_len], dim=0)
        speech_token_combined_emb = self.speech_embedding(speech_token_combined)

        # 3. prepare llm_input/target
        lm_target, lm_input, lm_input_len = self.prepare_lm_input_target(text_token.repeat(2, 1), text_token_emb.repeat(2, 1, 1), text_token_len.repeat(2),
                                                                         speech_token_combined, speech_token_combined_emb, speech_token_combined_len)
        lm_target = lm_target.to(device)

        # 4. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        chosen_logits = logits[:text_token.shape[0]]
        rejected_logits = logits[text_token.shape[0]:]
        chosen_lm_target = lm_target[:text_token.shape[0]]
        rejected_lm_target = lm_target[text_token.shape[0]:]
        loss = self.criterion_ce(chosen_logits, chosen_lm_target.to(device))
        acc = th_accuracy(chosen_logits.view(-1, self.speech_token_size + 3), chosen_lm_target, ignore_label=IGNORE_ID)

        # 5. calculate dpo logits
        chosen_lm_mask = chosen_lm_target == IGNORE_ID
        rejected_lm_mask = rejected_lm_target == IGNORE_ID
        chosen_logps = torch.gather(chosen_logits.log_softmax(dim=-1), dim=2, index=chosen_lm_target.masked_fill(chosen_lm_mask, 0).unsqueeze(dim=-1)).squeeze(dim=-1)
        rejected_logps = torch.gather(rejected_logits.log_softmax(dim=-1), dim=2, index=rejected_lm_target.masked_fill(rejected_lm_mask, 0).unsqueeze(dim=-1)).squeeze(dim=-1)
        chosen_logps = (chosen_logps * chosen_lm_mask).sum(dim=-1) / chosen_lm_mask.sum(dim=-1)
        rejected_logps = (rejected_logps * rejected_lm_mask).sum(dim=-1) / rejected_lm_mask.sum(dim=-1)
        return {'loss': loss, 'acc': acc, 'chosen_logps': chosen_logps, 'rejected_logps': rejected_logps}

    @torch.inference_mode()
    def inference(
            self,
            text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
    ):
        device = text.device
        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len
        
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=text.dtype).to(device)
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        # for token in self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid):
        #     yield token
        ret = self.inference_wrapper(lm_input, sampling, min_len, max_len, uuid)
        return ret

    @torch.inference_mode()
    def inference_wrapper(self, lm_input, sampling, min_len, max_len, uuid):

        out_tokens = []
        cache = None
        for i in range(max_len):
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                        masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
                                                        cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            #yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
        return out_tokens
    
    
    @torch.inference_mode()
    def inference_bistream(
            self,
            text: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            prompt_text_hidden:torch.Tensor = None,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
    ):

        device = prompt_text.device
        
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size, dtype=prompt_text.dtype).to(device)
        

        lm_input = torch.concat([sos_eos_emb], dim=1)
        
        text_cache = prompt_text
        # if prompt_text_hidden is None:
        #     prompt_text_hidden = torch.zeros_like(text_cache)
        # text_cache = text_cache + prompt_text_hidden
        
        out_tokens = []
        cache = None
        next_fill_index = -1

        for i in range(text.size(1)):
            this_text = text[:,i,:].clone().unsqueeze(0)
            text_cache = torch.concat([text_cache, this_text], dim=1)
            if i == text.size(1)-1:
                print('stop here')
            while prompt_speech_token_emb.size(1) != 0:
                if text_cache.size(1) >= self.mix_ratio[0]:
                    lm_input_text, lm_input_speech = text_cache[:, :self.mix_ratio[0]], prompt_speech_token_emb[:, :self.mix_ratio[1]]
                    logging.info('append {} text token {} speech token'.format(lm_input_text.size(1), lm_input_speech.size(1)))
                    lm_input = torch.concat([lm_input, lm_input_text, lm_input_speech], dim=1)
                    text_cache, prompt_speech_token_emb = text_cache[:, self.mix_ratio[0]:], prompt_speech_token_emb[:, self.mix_ratio[1]:]
                else:
                    logging.info('not enough text token to decode, wait for more')
                    break
            # no prompt_speech_token_emb remain, can decode some speech token
            if prompt_speech_token_emb.size(1) == 0:
                if (len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2) or (len(out_tokens) == 0 and lm_input.size(1) == 1):
                    logging.info('get fill token, need to append more text token')
                    if text_cache.size(1) >= self.mix_ratio[0]:
                        lm_input_text = text_cache[:, :self.mix_ratio[0]]
                        logging.info('append {} text token'.format(lm_input_text.size(1)))
                        if len(out_tokens) != 0 and out_tokens[-1] == self.speech_token_size + 2:
                            lm_input = lm_input_text
                        else:
                            lm_input = torch.concat([lm_input, lm_input_text], dim=1)
                        text_cache = text_cache[:, self.mix_ratio[0]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        continue
                while True:
                    seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
                    y_pred, cache = self.llm.forward_one_step(lm_input,
                                                              masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                              cache=cache)
                    logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                    if next_fill_index != -1 and len(out_tokens) == next_fill_index:
                        top_ids = self.speech_token_size + 2
                        next_fill_index += (self.mix_ratio[1] + 1)
                    else:
                        top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True).item()
                    if top_ids == self.speech_token_size + 2:
                        next_fill_index = len(out_tokens) + self.mix_ratio[1] + 1
                        logging.info('fill_token index {} next fill_token index {}'.format(len(out_tokens), next_fill_index))
                    
                    
                    out_tokens.append(top_ids)
                    if top_ids >= self.speech_token_size:
                        if top_ids == self.speech_token_size + 2:
                            break
                        else:
                            raise ValueError('should not get token {}'.format(top_ids))
                    # yield top_ids
                    lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        # 3. final decode
        if prompt_speech_token_emb.size(1)!= 0:
            lm_input = torch.concat([lm_input, text_cache, task_id_emb,prompt_speech_token_emb], dim=1)
        else:    
            lm_input = torch.concat([lm_input, text_cache, task_id_emb], dim=1)
        logging.info('no more text token, decode until met eos')
        while True:
            seq_len = lm_input.shape[1] if cache is None else lm_input.shape[1] + cache[0][0].size(2)
            y_pred, cache = self.llm.forward_one_step(lm_input,
                                                      masks=torch.tril(torch.ones((1, seq_len, seq_len), device=lm_input.device)).to(torch.bool),
                                                      cache=cache)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=False).item()
            out_tokens.append(top_ids)
            if top_ids >= self.speech_token_size:
                if top_ids == self.speech_token_size:
                    break
                else:
                    raise ValueError('should not get token {}'.format(top_ids))
            # in stream mode, yield token one by one
            # yield top_ids
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
        vocab_size = 6561

        print("最小 token id:", min(out_tokens))
        print("最大 token id:", max(out_tokens))

        negatives = [x for x in out_tokens if x < 0]
        overflows = [x for x in out_tokens if x >= vocab_size]
        middles = [x for x in out_tokens if 0<=x<vocab_size]
        print("小于 0 的数量:", len(negatives))
        print("大于等于 6561 的数量:", len(overflows))
        print("大于等于0，小于6561 的数量:", len(middles))
        if negatives:
            print("⚠️ 小于 0 的 token:", negatives[:20], "..." if len(negatives) > 20 else "")
        if overflows:
            print("⚠️ 大于等于 6561 的 token:", overflows[:20], "..." if len(overflows) > 20 else "")
        filtered_tokens = [t for t in out_tokens if t < 6561]
        return filtered_tokens
    

    

    @torch.inference_mode()
    def inference_reply_only(
        self,
        reply_mixed: torch.Tensor,         # [1, T_reply, D]
        prompt_hidden: torch.Tensor = None,
        sampling: int = 25,
        mix_ratio: tuple = (5, 15),        # (N_text, M_speech)
        sep_token_id: int = None,
        eos_token_id: int = None,
        max_decode: int = 4096,
    ):
        """
        规则：
        - 每当累积 >= N_text 的 reply 向量，就追加这 N_text 个，再“强制”解出 M_speech 个语音 token（忽略 EOS）。
        - 如果结尾剩余 < N_text，则不强制 15：把剩余直接追加，拼接 sep_token_id 后，正常 AR 直到 EOS。
        返回：List[int]，包含 sep 与 eos（如不需要可在外面去掉）
        """
        device = reply_mixed.device
        dtype  = reply_mixed.dtype
        N_text, M_speech = mix_ratio

        if eos_token_id is None:
            eos_token_id = self.speech_token_size
        if sep_token_id is None:
            sep_token_id = self.speech_token_size + 2

        # 1) 起始输入：<sos> + （可选）prompt_hidden
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1).to(device=device, dtype=dtype)
        sep_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1).to(device=device, dtype=dtype)
        if prompt_hidden is None:
            prompt_hidden = torch.zeros(1, 0, sos_eos_emb.size(-1), device=device, dtype=dtype)
        lm_input = torch.cat([prompt_hidden,sos_eos_emb], dim=1)
        cache = None

        out_tokens: list[int] = []
        total_steps_guard = 0

        # -- 单步前向 --
        def step_decode(lm_in: torch.Tensor, cache_state):
            seq_len = lm_in.shape[1] if cache_state is None else (lm_in.shape[1] + cache_state[0][0].size(2))
            causal = torch.tril(torch.ones((1, seq_len, seq_len), device=lm_in.device, dtype=torch.bool))
            y_pred, new_cache = self.llm.forward_one_step(lm_in, masks=causal, cache=cache_state)
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            return logp, new_cache

        # -- 追加一个有效 speech token 并转 embedding --
        def append_token_and_embed(token_id: int):
            nonlocal lm_input
            #if 0 <= token_id < self.speech_token_size:
            lm_input = self.speech_embedding.weight[token_id].reshape(1, 1, -1)
                
        # 2) 累积式地“吃” reply_mixed，并在每次凑满 N_text 时强制解出 M_speech
        T = reply_mixed.size(1)
        cursor = 0
        text_cache = torch.zeros(1, 0, sos_eos_emb.size(-1), device=device, dtype=dtype)

        while cursor < T:
            # 吃进一个位置
            chunk = reply_mixed[:, cursor:cursor+1, :]      # [1,1,D]
            text_cache = torch.cat([text_cache, chunk], dim=1)
            cursor += 1

            # 只在“凑满”N_text 时触发强制 M_speech
            while text_cache.size(1) >= N_text:
                # 2.1 追加 N_text 个 text 向量
                lm_input = torch.cat([lm_input, text_cache[:, :N_text, :]], dim=1)
                text_cache = text_cache[:, N_text:, :] #  clear out cache

                # 2.2 强制解出 M_speech（忽略 EOS）
                for _ in range(M_speech):
                    logp, cache = step_decode(lm_input, cache)
                    next_id = self.sampling_ids(
                        logp.squeeze(0), out_tokens, sampling, ignore_eos=True
                    ).item()

                    out_tokens.append(next_id)
                    append_token_and_embed(next_id)

                    total_steps_guard += 1
                    if total_steps_guard >= max_decode:
                        import logging
                        logging.warning("reach max_decode during forced chunk; early return")
                        return out_tokens

        # 3) 走到这里说明 reply_mixed 已吃完
        #    若 text_cache 里还有 < N_text 的残余：不再强制 15，直接加进去即可
        if text_cache.size(1) > 0:
            lm_input = torch.cat([lm_input, text_cache, sep_emb], dim=1)

        # 4) 正常自回归到 EOS
        while True:
            logp, cache = step_decode(lm_input, cache)
            next_id = self.sampling_ids(
                logp.squeeze(0), out_tokens, sampling, ignore_eos=False
            ).item()

            if next_id == eos_token_id:
                break
            out_tokens.append(next_id)
            
            append_token_and_embed(next_id)

            total_steps_guard += 1
            if total_steps_guard >= max_decode:
                import logging
                logging.warning("reach max_decode during tail AR; force append EOS and stop")
                if out_tokens[-1] != eos_token_id:
                    out_tokens.append(eos_token_id)
                break

        return out_tokens