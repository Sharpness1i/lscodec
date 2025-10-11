from loss import *
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import abstractmethod
from dataclasses import dataclass
import logging
import typing as tp
import torch
from torch import nn
from model.model import MultiPeriodDiscriminator, MultiResolutionDiscriminator, DACDiscriminator
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AutoModel
import os
WAVLM_DIR= os.environ.get("WAVLM_DIR") 

from ..quantization import (
    QuantizedResult,
    BaseQuantizer,
    SplitResidualVectorQuantizer,
    ResidualVectorQuantizer,
)
from ..modules.conv import pad_for_conv1d
from ..modules.resample import ConvDownsample1d, ConvTrUpsample1d
from ..modules.streaming import StreamingModule, State, StateT
from ..utils.compile import CUDAGraphed
from model.criterion import ContrasiveCriterion, DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, DACGANLoss, MelSpecReconstructionLoss, STFTSpecReconstructionLoss

logger = logging.getLogger()


class CompressionModel(StreamingModule[StateT]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
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
class _MimiState(State):
    graphed_tr_enc: CUDAGraphed | None
    graphed_tr_dec: CUDAGraphed | None
    graphed_encoder: CUDAGraphed
    graphed_decoder: CUDAGraphed


import json
with open('/root/code/lscodec/conf/spt_base_cfg.json') as f:
    cfg = json.load(f)

class MimiModel(pl.LightningModule, CompressionModel[_MimiState]):
    """Mimi model operating on the raw waveform.

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
    ):
        super().__init__()
        
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresdisc = MultiResolutionDiscriminator()
        self.dac = DACDiscriminator()
        
        self.dacdiscriminator = DACGANLoss(self.dac)
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=16000)
        self.stftspec_loss = STFTSpecReconstructionLoss() 
        self.teacher_feature_extractor = AutoModel.from_pretrained(WAVLM_DIR).eval()
        self.teacher_feature_extractor.requires_grad_(False)
        
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.cfg=cfg
        self.mel_loss_lambdas = cfg.get('mel_loss_lambdas')
        self.commitment_loss_lambda = cfg.get('commitment_loss_lambda')
        self.recon_loss_lambda = cfg.get('recon_loss_lambda')
 
        
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
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
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
        self.mel_loss_kwargs_list = []
        mult = 1
        for i in range(len(self.mel_loss_lambdas)):
            self.mel_loss_kwargs_list.append({'n_fft': cfg.get('n_fft') // mult, 'num_mels':cfg.get('num_mels'),'sample_rate':self.sample_rate,
                                 'hop_size': cfg.get('hop_size') // mult, 'win_size':cfg.get('win_size') // mult, 'fmin':cfg.get('fmin'), 
                                'fmax':cfg.get('fmax_for_loss')})
            mult = mult * 2
        self.mel_kwargs = {'n_fft': cfg.get('n_fft'), 'num_mels':cfg.get('num_mels'),'sample_rate':self.sample_rate,
                                 'hop_size': cfg.get('hop_size'), 'win_size':cfg.get('win_size'), 'fmin':cfg.get('fmin'), 
                                'fmax':cfg.get('fmax')}
        self.automatic_optimization = False

    def _init_streaming_state(self, batch_size: int) -> _MimiState:
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
        return _MimiState(batch_size, device, graphed_tr_enc, graphed_tr_dec, graphed_encoder, graphed_decoder)

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
            {"params": self.encoder_transformer.parameters()},
            {"params": self.decoder_transformer.parameters()},
        ]
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresdisc.parameters()},
            {"params": self.dac.parameters()},
        ]
        opt_gen = torch.optim.AdamW(gen_params, self.cfg['opt_gen_lr'])
        opt_disc = torch.optim.AdamW(disc_params, self.cfg['opt_disc_lr'])

        sch_gen = get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.cfg['sch_warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )
        sch_disc = get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.cfg['sch_warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )
        return [opt_gen, opt_disc], [sch_gen, sch_disc]
    


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

        emb = self.encoder(x)

        if self.encoder_transformer is not None:
            (emb,) = self.encoder_transformer(emb)
        emb = self._to_framerate(emb)
        expected_length = self.frame_rate * length / self.sample_rate
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
        
        return recon , loss, q_res.distill_loss
    
    def training_step(self,batch,batch_idx):
        total_mel_error = 0
        wav_16k, wav_24k, lengths_16k, lengths_24k = batch
        # teacher model input wav has to be 16k
        teacher_feature = self.teacher_feature_extractor(wav_16k).last_hidden_state.detach() # (b,t,1024)
        if wav_24k.dim() == 2:
            wav_24k = wav_24k.unsqueeze(1)
            
        state = self._streaming_state
        frame_size = self.frame_size 
        if state is None:
            wav = pad_for_conv1d(wav_24k, frame_size, frame_size)
        
        ######################## discriminator ##############################
        opt_gen, opt_disc = self.optimizers()
        sch_gen, sch_disc = self.lr_schedulers()

        # discriminator step
        with torch.no_grad():
            x_hat, _, _ = self(wav,teacher_feature=teacher_feature)
        
        loss_dac = self.dacdiscriminator.discriminator_loss(x_hat, wav)
        
        real_score_mpd, gen_score_mpd, _, _ = self.multiperioddisc(y=wav.squeeze(1), y_hat=x_hat.squeeze(1))
        real_score_mrd, gen_score_mrd, _, _ = self.multiresdisc(y=wav.squeeze(1), y_hat=x_hat.squeeze(1))
        loss_mpd, loss_mpd_real, _ = self.disc_loss(
           disc_real_outputs=real_score_mpd, disc_generated_outputs=gen_score_mpd
        )
        loss_mrd, loss_mrd_real, _ = self.disc_loss(
           disc_real_outputs=real_score_mrd, disc_generated_outputs=gen_score_mrd
        )
        loss_mpd /= len(loss_mpd_real)
        loss_mrd /= len(loss_mrd_real)
        loss_disc = loss_mpd + loss_mrd + loss_dac

        opt_disc.zero_grad()
        self.manual_backward(loss_disc)
        self.clip_gradients(opt_disc, gradient_clip_val=self.config['gradient_clip_val'], gradient_clip_algorithm='norm')
        opt_disc.step()
        sch_disc.step()
        self.log("train/loss_disc", float(loss_disc), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_mpd", float(loss_mpd), on_step=True, on_epoch=True)
        self.log("train/loss_mrd", float(loss_mrd), on_step=True, on_epoch=True)
        self.log("train/loss_dac", float(loss_dac), on_step=True, on_epoch=True)
        
        
        ######################### generator ####################################################
        x_hat, commit_loss, distill_loss = self(wav,teacher_feature=teacher_feature)
             
        mel_error = mel_loss(wav, x_hat, **self.mel_loss_kwargs_list[0]).item()
        total_mel_error += mel_error
         
        loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(x_hat, wav)  # 完全没有平均
        _, gen_score_mpd, fmap_rs_mpd, fmap_gs_mpd = self.multiperioddisc(
           y=wav.squeeze(1), y_hat=x_hat.squeeze(1),
        )
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresdisc(
           y=wav.squeeze(1), y_hat=x_hat.squeeze(1),
        )
        loss_gen_mpd, list_loss_gen_mpd = self.gen_loss(disc_outputs=gen_score_mpd)
        loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
        loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)  # 每个子disc平均
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)  # 每个子disc平均
        loss_fm_mpd = self.feat_matching_loss(fmap_r=fmap_rs_mpd, fmap_g=fmap_gs_mpd) / len(fmap_rs_mpd)  # 每个子disc平均
        loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)  # 每个子disc平均

        loss_adv = loss_dac_1 + loss_gen_mpd + loss_gen_mrd
        loss_fm = loss_dac_2 + loss_fm_mpd + loss_fm_mrd

        loss_mel = self.melspec_loss(x_hat, wav)

        if self.use_perceptual_loss:
            perceptual_loss = self.cal_perceptual_loss(x_hat, wav)
            p_weight = self.calculate_adaptive_weight(
                loss_mel * 45,
                perceptual_loss,
                self.get_last_layer(),
            )
            # p_weight = 1.0
        else:
            perceptual_loss = 0.0
            p_weight = 0.0

        
        d_weight = self.calculate_adaptive_weight(
            loss_mel * 45,
            loss_adv + loss_fm,
            self.get_last_layer(),
        )
        # d_weight = 1.0

        loss_gen = d_weight * (loss_adv + loss_fm) + 45 * loss_mel + commit_loss + distill_loss  + p_weight * perceptual_loss
        print(commit_loss)
        
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(opt_gen, gradient_clip_val=self.config['gradient_clip_val'], gradient_clip_algorithm='norm')
        opt_gen.step()
        sch_gen.step()
        
        self.log("train/loss_gen", float(loss_gen), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_adv", float(loss_adv), on_step=True, on_epoch=True)
        self.log("train/loss_fm", float(loss_fm), on_step=True, on_epoch=True)
        self.log("train/loss_mel", float(loss_mel), on_step=True, on_epoch=True)
        self.log("train/perceptual_loss", float(perceptual_loss), on_step=True, on_epoch=True)
        self.log("train/loss_semantic", float(distill_loss), on_step=True, on_epoch=True)
        self.log("train/commit_loss", float(commit_loss), on_step=True, on_epoch=True)
        self.log("train/d_weight", float(d_weight), on_step=True, on_epoch=True)
        self.log("train/p_weight", float(p_weight), on_step=True, on_epoch=True)

        self.current_training_step += 1

        pass

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
                    "You are responsible for buffering accordingly before feeding audio to Mimi.")
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
