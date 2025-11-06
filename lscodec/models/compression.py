from loss import *
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import abstractmethod
from dataclasses import dataclass
import logging
import typing as tp
import torch
from torch import nn
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoModel
import os
import soundfile as sf
from losses import disc_loss, total_loss
WAVLM_DIR= os.environ.get("WAVLM_DIR") 
from ..quantization import (QuantizedResult,BaseQuantizer,SplitResidualVectorQuantizer,ResidualVectorQuantizer)
from ..modules.conv import pad_for_conv1d
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.streaming import StreamingModule, State, StateT
from ..utils.compile import CUDAGraphed
from model.criterion import ContrasiveCriterion
logger = logging.getLogger()



class CompressionModel(StreamingModule[StateT]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `lscodecModel.encode`."""
        ...
    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `lscodecModel.decode`."""
        ...
    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...
    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_size(self) -> int: ...
    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...


@dataclass
class _lscodecState(State):
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None
    graphed_encoder: CUDAGraphed
    graphed_decoder: CUDAGraphed



class lscodecModel(pl.LightningModule, CompressionModel[_lscodecState]):
    """lscodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (float): Final frame rate of the quantized representatiopn.
        encoder_frame_rate (float): frame rate of the encoder model. Note that if `frame_rate != encopder_frame_rate`,
            the latent will be resampled linearly to match the desired `frame_rate` before and after quantization.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        encoder_transformer (nn.Module or None): optional transformer for the encoder.
        decoder_transformer (nn.Module or None): optional transformer for the decoder.
        resample_method (str): method to use for resampling the latent space before the quantizer.
        upsample_channel_wise_bug (bool): controls whether the upsampling is channel wise.
            Defaults to true to reproduce bug in original implementation.
        freeze_encoder: whether to freeze the encoder weights.
        freeze_quantizer: whether to freeze the quantizer weights.
        freeze_quantizer_level: If positive, freeze the quantizer up to this level.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        config: str = None,
        recon_dir: str = None,
        discriminator_start_step: int = 10000,
    ):
        self.recon_dir = recon_dir
        super().__init__()
        self.disc_model = MultiScaleSTFTDiscriminator(filters=32)
        self.teacher_feature_extractor = AutoModel.from_pretrained(WAVLM_DIR).eval()
        self.teacher_feature_extractor.eval()
        for p in self.teacher_feature_extractor.parameters():
            p.requires_grad = False
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.automatic_optimization = False
        
        self.loss_lambda = {
            "adv_genloss": 2.0,          # 对应 losses_g['l_g']  # 1-3    细粒度 补充
            "l_feat": 10.0,               # 对应 losses_g['l_feat']    #  中间层次 ；粗粒度
            "waveform_loss": 0.1,        # 对应 losses_g['l_t']  # 0.1   细粒度
            "ms_mel_loss": 15.0,          # 对应 losses_g['l_f'] # 粗粒度 补充
            "commit_loss": 1.0,          # 对应 commit_loss
            "distill_loss": 10.0,        # 对应 distill_loss
        }
   
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.encoder_transformer is not None:
                for p in self.encoder_transformer.parameters():
                    p.requires_grad = False
            for name, p in self.quantizer.named_parameters():
                if name.endswith("input_proj.weight"):
                    p.requires_grad = False
        if freeze_quantizer:
            self.quantizer.ema_frozen_(True)
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        self.total_steps = 0
        self.discriminator_start_step= discriminator_start_step
        assert resample_method in ["interpolate","conv","avg_pool",], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (causal and resample_method == "interpolate"), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (self.encoder_frame_rate > self.frame_rate), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(downsample_stride), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                if freeze_encoder:
                    for p in self.downsample.parameters():
                        p.requires_grad = False
                self.upsample = ConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )


    def _init_streaming_state(self, batch_size: int) -> _lscodecState:
        device = next(self.parameters()).device
        disable = device.type != 'cuda'
        graphed_tr_dec = None
        graphed_tr_enc = None
        if self.encoder_transformer is not None:
            graphed_tr_enc = CUDAGraphed(self.encoder_transformer, disable=disable)
        if self.decoder_transformer is not None:
            graphed_tr_dec = CUDAGraphed(self.decoder_transformer, disable=disable)
        graphed_encoder = CUDAGraphed(self.encoder, disable=disable)
        graphed_decoder = CUDAGraphed(self.decoder, disable=disable)
        return _lscodecState(batch_size, device, graphed_tr_enc, graphed_tr_dec, graphed_encoder, graphed_decoder)

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available."""
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer."""
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook."""
        return self.quantizer.cardinality

    def _to_framerate(self, x: torch.Tensor):
        # Convert from the encoder frame rate to the overall framerate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.downsample(x)

    def _to_encoder_framerate(self, x: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        _, _, length = x.shape
        frame_rate = self.encoder_frame_rate
        new_frame_rate = self.frame_rate
        if frame_rate == new_frame_rate:
            return x
        if self.resample_method == "interpolate":
            target_length = int(length * new_frame_rate / frame_rate)
            return nn.functional.interpolate(x, size=target_length, mode="linear")
        else:
            return self.upsample(x)
    

    def configure_optimizers(self):
        
        gen_params = [
        {"params": self.encoder.parameters()},
        {"params": self.decoder.parameters()},
        {"params": self.quantizer.parameters()},
        ]
        if self.encoder_transformer is not None:
            gen_params.append({"params": self.encoder_transformer.parameters()})
        if self.decoder_transformer is not None:
            gen_params.append({"params": self.decoder_transformer.parameters()})
        disc_params = [{"params": self.disc_model.parameters()}]

        opt_gen = torch.optim.AdamW(gen_params, self.config['opt_gen']['lr'])
        opt_disc = torch.optim.AdamW(disc_params, self.config['opt_disc']['lr'])

        sch_gen = get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.config['sch']['warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )
        sch_disc = get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.config['sch']['warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )
        return [opt_gen, opt_disc], [sch_gen, sch_disc]
    
    
    def get_training_progress(self, batch):
        # 当前全局步数（Lightning 已自动同步）
        global_step = self.global_step
        epoch = self.current_epoch

        # 当前 batch 大小（一个 GPU 上的 batch）
        batch_size = batch[0].size(0) if isinstance(batch, (tuple, list)) else batch.size(0)

        # world_size（多 GPU / 多节点）
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1

        # 已处理样本数（全局视角）
        total_samples = global_step * batch_size * world_size

        return {
            "global_step": global_step,
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "total_samples": total_samples
        }
    

    def forward(self, x ,teacher_feature=None) -> QuantizedResult:
        length = x.shape[-1]
        extra_metrics: tp.Dict[str, torch.Tensor] = {}
        if self.freeze_quantizer:
            if isinstance(self.quantizer, SplitResidualVectorQuantizer):
                self.quantizer.rvq_first.eval()
                for i in range(
                    self.freeze_quantizer_level - self.quantizer.n_q_semantic
                ):
                    self.quantizer.rvq_rest.vq.layers[i].eval()
            elif isinstance(self.quantizer, ResidualVectorQuantizer):
                for i in range(self.freeze_quantizer_level):
                    self.quantizer.vq.layers[i].eval()
            else:
                raise ValueError(f"Unsupported quantizer type {type(self.quantizer)}")

        emb = self.encoder(x) # 960倍下采样

        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb) # 时间长度不变
        emb = self._to_framerate(emb) # 下采样 2倍 ；
        expected_length = self.frame_rate * length / self.sample_rate  # 这个expected length 正是 samples // 1920  (符合 frame_rate = 12.5HZ 的长度)
        # Checking that we have the proper length given the advertised frame rate.
        assert abs(emb.shape[-1] - expected_length) < 1, (
            emb.shape[-1],
            expected_length,
        )
        q_res = self.quantizer(emb,teacher_feature,self.frame_rate)
        loss = q_res.penalty
        emb = q_res.x
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            (emb,) = self.decoder_transformer(emb)

        out = self.decoder(emb)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        recon = out[..., :length]

        q_res.x = out
        q_res.metrics.update(extra_metrics)
        
        return recon,loss,q_res.distill_loss
        
    def on_fit_start(self):
        self.total_steps = 0
        

    def training_step(self, batch, batch_idx):
        wav_16k, wav_24k, lengths_16k, lengths_24k = batch['speech_16k'], batch['speech'], batch['speech_16k_lens'], batch['speech_lens']
        
        teacher_feature = self.teacher_feature_extractor(wav_16k).last_hidden_state.detach()
 
        if wav_24k.dim() == 2:
            wav_24k = wav_24k.unsqueeze(1)
        wav = pad_for_conv1d(wav_24k, self.frame_size, self.frame_size)

        opt_gen, opt_disc = self.optimizers()
        sch_gen, sch_disc = self.lr_schedulers()

        self.total_steps += 1
        if torch.distributed.is_initialized():
            t = torch.tensor(self.total_steps, device=self.device)
            torch.distributed.broadcast(t, src=0)
            self.total_steps = int(t.item())
            
        x_hat, commit_loss, distill_loss = self(wav, teacher_feature=teacher_feature)

        # ===== 判别器 =====
        opt_disc.zero_grad()
        loss_disc = torch.tensor(0.0, device=self.device)
        
        logits_real, fmap_real = self.disc_model(wav)
        logits_fake, fmap_fake = self.disc_model(x_hat.detach())
        loss_disc = disc_loss(logits_real, logits_fake)
        self.manual_backward(loss_disc)
        self.clip_gradients(opt_disc, 5.0, "norm")
        opt_disc.step()
        sch_disc.step()

        opt_gen.zero_grad()
        logits_real, fmap_real = self.disc_model(wav)
        logits_fake, fmap_fake = self.disc_model(x_hat)
        losses_g = total_loss(
            fmap_real=fmap_real,
            logits_fake=logits_fake,
            fmap_fake=fmap_fake,
            input_wav=wav,
            output_wav=x_hat,
            sample_rate=24000,
        )

        loss_gen = (
            self.loss_lambda["adv_genloss"] * losses_g["l_g"]
            + self.loss_lambda["l_feat"] * losses_g["l_feat"]
            + self.loss_lambda["waveform_loss"] * losses_g["l_t"]
            + self.loss_lambda["ms_mel_loss"] * losses_g["l_f"]
            + self.loss_lambda["commit_loss"] * commit_loss
            + self.loss_lambda["distill_loss"] * distill_loss
        )

        self.manual_backward(loss_gen)
        self.clip_gradients(opt_gen, 5.0, "norm")
        opt_gen.step()
        sch_gen.step()

        log_dict = {
            "train/generator_total_loss": loss_gen.detach(),
            "train/adv_genloss": self.loss_lambda["adv_genloss"] * losses_g["l_g"].detach(),
            "train/l_feat": self.loss_lambda["l_feat"] * losses_g["l_feat"].detach(),
            "train/waveform_loss": self.loss_lambda["waveform_loss"] * losses_g["l_t"].detach(),
            "train/ms_mel_loss": self.loss_lambda["ms_mel_loss"] * losses_g["l_f"].detach(),
            "train/commit_loss": self.loss_lambda["commit_loss"] * commit_loss.detach(),
            "train/distill_loss": self.loss_lambda["distill_loss"] * distill_loss.detach(),
            "train/loss_disc": loss_disc.detach(),
        }

        self.log_dict(log_dict, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        
        if (self.total_steps % 10 == 0) or (batch_idx == 0):
            print(f"\n[Step {self.total_steps:>6d}] | ", end="")

            console_order = [
                "train/generator_total_loss",
                "train/loss_disc",
                "train/adv_genloss",
                "train/l_feat",
                "train/waveform_loss",
                "train/ms_mel_loss",
                "train/commit_loss",
                "train/distill_loss",
            ]

            console_str = " | ".join(
                [f"{name.split('/')[-1]}: {log_dict[name].item():8.3f}" for name in console_order]
            )
            print(console_str, flush=True)

        return {"loss_gen": loss_gen.detach(), "loss_disc": loss_disc.detach()}
        
    def test_step(self, batch):
        wav_type, waveform, fs, length, postfix = batch
        codes = self.encode(waveform.unsqueeze(1))
        reconstructed_wav = self.decode(codes)

        base, ext = os.path.splitext(postfix[0])
        recon_name = f"{base}_recon{ext}"
        os.makedirs(self.recon_dir, exist_ok=True)
        save_path = os.path.join(self.recon_dir, recon_name)
        fs = int(fs.item()) if isinstance(fs, torch.Tensor) else int(fs)

        wav_to_save = reconstructed_wav.squeeze().detach().cpu().numpy()

        if wav_to_save.ndim == 3:
            wav_to_save = wav_to_save[0]
        
        if wav_to_save.ndim == 1:
            wav_to_save = wav_to_save.unsqueeze(0)
        
        torchaudio.save(save_path, wav_to_save, sample_rate=fs)

        print(f"Saved reconstructed wav to: {save_path}")


    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a batch of waveforms to unquantized latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Unquantized embeddings.
        """
        assert (
            x.dim() == 3
        ), f"CompressionModel._encode_to_unquantized_latent expects audio of shape [B, C, T] but got {x.shape}"

        state = self._streaming_state
        frame_size = self.frame_size

        if state is None:
            # The underlying convolutions no longer accept partial inputs,
            # `x` needs to be exactly a multiple of the frame size,
            # reproducing the previous padding behavior here.
            x = pad_for_conv1d(x, frame_size, frame_size)
            emb = self.encoder(x)
        else:
            if x.shape[-1] % frame_size != 0 or x.shape[-1] == 0:
                raise RuntimeError(
                    f"Invalid input x of length {x.shape[-1]}. The length must be "
                    f"a positive multiple of the frame size {frame_size}. "
                    "You are responsible for buffering accordingly before feeding audio to lscodec.")
            emb = state.graphed_encoder(x).clone()
        if self.encoder_transformer is not None:
            if state is None:
                (emb,) = self.encoder_transformer(emb)
            else:
                assert state.graphed_tr_enc is not None
                (emb,) = state.graphed_tr_enc(emb)
        emb = self._to_framerate(emb)
        return emb

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the given input tensor to quantized representation.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes (torch.Tensor): an int tensor of shape [B, K, T]
                with K the number of codebooks used and T the timestep.
        """
        emb = self._encode_to_unquantized_latent(x)
        codes = self.quantizer.encode(emb)
        return codes

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        """Projects a batch of waveforms to latent space.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T].

        Returns:
            Embeddings, either quantized or not.
        """
        emb = self._encode_to_unquantized_latent(x)
        if not quantize:
            return emb
        else:
            codes = self.quantizer.encode(emb)
            return self.decode_latent(codes)

    def decode(self, codes: torch.Tensor):
        """Decode the given codes to a reconstructed representation.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        state = self._streaming_state
        emb = self.decode_latent(codes)
        emb = self._to_encoder_framerate(emb)
        if self.decoder_transformer is not None:
            if state is None:
                (emb,) = self.decoder_transformer(emb)
            else:
                assert state.graphed_tr_dec is not None
                (emb,) = state.graphed_tr_dec(emb)
        if state is None:
            out = self.decoder(emb)
        else:
            out = state.graphed_decoder(emb).clone()
        # out contains extra padding added by the encoder and decoder
        return out

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        return self.quantizer.decode(codes)


class WrapperCompressionModel(CompressionModel[State]):
    """Base API for CompressionModel wrappers that do not depend on external frameworks."""

    def __init__(self, model: CompressionModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> QuantizedResult:
        return self.model.forward(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode_latent(codes)

    def set_num_codebooks(self, n: int):
        self.model.set_num_codebooks(n)

    @property
    def quantizer(self):
        return self.model.quantizer

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def frame_rate(self) -> float:
        return self.model.frame_rate

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def frame_size(self) -> int:
        return self.model.frame_size

    @property
    def cardinality(self) -> int:
        return self.model.cardinality

    @property
    def num_codebooks(self) -> int:
        return self.model.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.model.total_codebooks
