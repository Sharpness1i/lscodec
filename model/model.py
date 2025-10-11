import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import time
import math
import random
from pesq import pesq
import soundfile as sf
from torchaudio.transforms import MelSpectrogram
from matplotlib import pyplot as plt
import transformers


import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
HUBERT_DIR= os.environ.get("HUBERT_DIR") 

from .vq import Codec
from .disc import MultiPeriodDiscriminator, MultiResolutionDiscriminator, DACDiscriminator
from .criterion import ContrasiveCriterion, DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, DACGANLoss, MelSpecReconstructionLoss, STFTSpecReconstructionLoss


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.generator = Codec(config['encoder_config'], config['decoder_config'], config['quantizer_config'])
        self.multiperioddisc = MultiPeriodDiscriminator()
        self.multiresdisc = MultiResolutionDiscriminator()
        self.dac = DACDiscriminator()
        self.dacdiscriminator = DACGANLoss(self.dac)

        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=16000)
        self.stftspec_loss = STFTSpecReconstructionLoss()  # 仅用来计算验证集指标
        # self.contrasive_loss = ContrasiveCriterion(**config['contrasive_criterion_config'])

        self.utmos_model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        )

        if 'pretrained_pt_path' in config and config['pretrained_pt_path'] is not None:
            state_dict = torch.load(config['pretrained_pt_path'], map_location='cpu')
            del state_dict['generator.semantic_encoder.conv.conv.weight']
            del state_dict['generator.semantic_decoder.conv2.conv.weight']
            self.load_state_dict(state_dict, strict=False)


        self.feature_extractor = AutoModel.from_pretrained(HUBERT_DIR).eval()
        self.feature_extractor.requires_grad_(False)

        self.current_training_step = -1
        self.automatic_optimization = False
    

    # 重写 state_dict: 排除 utmos_model 和 feature_extractor
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith('utmos_model.') or key.startswith('feature_extractor.'):
                del state[key]
        return state

    # 重写 load_state_dict: 排除 utmos_model 和 feature_extractor
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=False)

    
    def pad_wav(self, wav):
        hop_length = math.prod(self.config['encoder_config']['ratios']) * 2
        pad_length = math.ceil(wav.size(-1) / hop_length) * hop_length - wav.size(-1)
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        return wav

    @torch.no_grad()
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # wavs: (b,t)
        wavs = F.pad(wavs, (160, 160))
        
        feats = self.feature_extractor(wavs, output_hidden_states=True)
        feats_mix = torch.stack(feats.hidden_states, dim=1).mean(1)

        # 进行幅度压缩！！！
        symbol = (feats_mix > 0).float() * 2 - 1
        magnitude = feats_mix.abs() ** 0.3
        feats_mix = symbol * magnitude

        return feats_mix
    

    def forward(self, wav, domain_split=None):
        # wav: (b,t)
        feat = self.extract_wav2vec2_features(wav).transpose(-2, -1).detach()  # (b,d,t)
        recon, pred_feat, commit_loss = self.generator(wav.unsqueeze(1), feat, domain_split=domain_split)
        return recon, pred_feat, feat, commit_loss


    def on_train_batch_start(self, *args):
        if self.current_training_step >= self.config['perceptual_start_step']:
            self.use_perceptual_loss = True
        else:
            self.use_perceptual_loss = False
    

    def training_step(self, batch, batch_idx):
        wav, fs, lengths, names = batch

        wav = self.pad_wav(wav)  # [b,t]

        opt_gen, opt_disc = self.optimizers()
        sch_gen, sch_disc = self.lr_schedulers()

        ############################### generator step #############################################
        # recon, commit_loss, cnn_feat, mask_indices, quantized = self(wav, use_mask=use_mask)
        recon, pred_feat, feat, commit_loss = self(wav)

        loss_dac_1, loss_dac_2 = self.dacdiscriminator.generator_loss(recon.unsqueeze(1), wav.unsqueeze(1))  # 完全没有平均
        _, gen_score_mpd, fmap_rs_mpd, fmap_gs_mpd = self.multiperioddisc(
            y=wav, y_hat=recon,
        )
        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = self.multiresdisc(
            y=wav, y_hat=recon,
        )
        loss_gen_mpd, list_loss_gen_mpd = self.gen_loss(disc_outputs=gen_score_mpd)
        loss_gen_mrd, list_loss_gen_mrd = self.gen_loss(disc_outputs=gen_score_mrd)
        loss_gen_mpd = loss_gen_mpd / len(list_loss_gen_mpd)  # 每个子disc平均
        loss_gen_mrd = loss_gen_mrd / len(list_loss_gen_mrd)  # 每个子disc平均
        loss_fm_mpd = self.feat_matching_loss(fmap_r=fmap_rs_mpd, fmap_g=fmap_gs_mpd) / len(fmap_rs_mpd)  # 每个子disc平均
        loss_fm_mrd = self.feat_matching_loss(fmap_r=fmap_rs_mrd, fmap_g=fmap_gs_mrd) / len(fmap_rs_mrd)  # 每个子disc平均

        loss_adv = loss_dac_1 + loss_gen_mpd + loss_gen_mrd
        loss_fm = loss_dac_2 + loss_fm_mpd + loss_fm_mrd

        loss_mel = self.melspec_loss(recon, wav)

        loss_semantic = F.mse_loss(pred_feat, feat)

        if self.use_perceptual_loss:
            perceptual_loss = self.cal_perceptual_loss(recon, wav)
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

        loss_gen = d_weight * (loss_adv + loss_fm) + 45 * loss_mel + commit_loss + loss_semantic + p_weight * perceptual_loss

        
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
        self.log("train/loss_semantic", float(loss_semantic), on_step=True, on_epoch=True)
        self.log("train/commit_loss", float(commit_loss), on_step=True, on_epoch=True)
        self.log("train/d_weight", float(d_weight), on_step=True, on_epoch=True)
        self.log("train/p_weight", float(p_weight), on_step=True, on_epoch=True)

        self.current_training_step += 1
        
        
        
        ##################################### discriminator step#############################################
        with torch.no_grad():
            recon, _, _, _ = self(wav)
        
        loss_dac = self.dacdiscriminator.discriminator_loss(recon.unsqueeze(1), wav.unsqueeze(1))
        real_score_mpd, gen_score_mpd, _, _ = self.multiperioddisc(y=wav, y_hat=recon)
        real_score_mrd, gen_score_mrd, _, _ = self.multiresdisc(y=wav, y_hat=recon)
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

        # visualize
        if self.global_rank == 0 and self.current_training_step % 400 == 0:
            self.logger.experiment.add_audio("train/audio_in", wav[0].data.cpu(), self.current_training_step, 16000)
            self.logger.experiment.add_audio("train/audio_pred", recon[0].data.cpu(), self.current_training_step, 16000)

            mel_target = self.melspec_loss.mel_spec(wav[0]).clamp(1e-5).log10()
            mel_hat = self.melspec_loss.mel_spec(recon[0]).clamp(1e-5).log10()
            self.logger.experiment.add_image(
                "train/mel_target",
                self.plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.current_training_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "train/mel_hat",
                self.plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.current_training_step,
                dataformats="HWC",
            )

    def validation_step(self, batch, batch_idx):
        domain, wav, fs, lengths, names = batch
        assert wav.size(0) == 1

        recon, _, _, _ = self(self.pad_wav(wav))
        tgt, est = wav, recon[..., :wav.size(-1)]  # (b,t)

        if self.global_rank == 0 and batch_idx == 0:
            self.logger.experiment.add_audio("valid/audio_in", tgt[0].data.cpu(), self.current_training_step, 16000)
            self.logger.experiment.add_audio("valid/audio_pred", est[0].data.cpu(), self.current_training_step, 16000)

            mel_target = self.melspec_loss.mel_spec(tgt[0]).clamp(1e-5).log10()
            mel_hat = self.melspec_loss.mel_spec(est[0]).clamp(1e-5).log10()
            self.logger.experiment.add_image(
                "valid/mel_target",
                self.plot_spectrogram_to_numpy(mel_target.data.cpu().numpy()),
                self.current_training_step,
                dataformats="HWC",
            )
            self.logger.experiment.add_image(
                "valid/mel_hat",
                self.plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                self.current_training_step,
                dataformats="HWC",
            )
        
        domain = domain[0]
        if domain == 'speech':
            utmos_score = float(self.utmos_model(est, 16000).cpu().item())
            tgt = tgt.squeeze().cpu().numpy()
            est = est.squeeze().cpu().numpy()
            try:
                pesq_score = pesq(16000, tgt, est, 'wb')
            except:
                pesq_score = 1.0
            
            if pesq_score is not None:
                self.log_dict({'valid_speech/pesq_score': float(pesq_score)}, on_step=False, on_epoch=True, sync_dist=True)
            self.log("valid_speech/utmos_score", utmos_score, on_step=False, on_epoch=True, sync_dist=True)
        else:
            mel_dist = self.melspec_loss(est, tgt)
            stft_dist = self.stftspec_loss(est, tgt)
            self.log_dict({f'valid_{domain}/mel_dist': float(mel_dist)}, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict({f'valid_{domain}/stft_dist': float(stft_dist)}, on_step=False, on_epoch=True, sync_dist=True)    

        

    def on_validation_epoch_end(self,):
        # save ckpt when validation epoch finished
        if self.trainer.sanity_checking:
            return
        epoch = self.current_epoch
        step = self.current_training_step
        pesq_score = self.trainer.callback_metrics['valid_speech/pesq_score']
        utmos_score = self.trainer.callback_metrics['valid_speech/utmos_score']
        ckpt_name = f'epoch={epoch}-step={step}-pesq={pesq_score:.2f}-utmos={utmos_score:.2f}.ckpt'
        self.trainer.save_checkpoint(self.config['ckpt_dir'] / ckpt_name)
    
    @staticmethod
    def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
        """
        Plot a spectrogram and convert it to a numpy array.

        Args:
            spectrogram (ndarray): Spectrogram data.

        Returns:
            ndarray: Numpy array representing the plotted spectrogram.
        """
        spectrogram = spectrogram.astype(np.float32)
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        try:
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except:
            # 使用 tostring_argb 并处理 ARGB 格式
            data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换 ARGB 到 RGB
            data = data[:, :, 1:]  # 去掉 alpha 通道
        plt.close()
        return data

    
    def test_step(self, batch, batch_idx):

        wav, fs, lengths, names = batch

        recon, _, _, _ = self(self.pad_wav(wav))
        est = recon[..., :wav.size(-1)].squeeze().cpu().numpy()

        if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
            sf.write(Path(self.config['save_enhanced']) / f'{names[0]}', est, samplerate=int(fs[0]))
        
    
    def on_test_epoch_end(self,):
        pass


    def on_save_checkpoint(self, ckpt):
        ckpt['current_training_step'] = self.current_training_step
    
    def on_load_checkpoint(self, ckpt):
        self.current_training_step = ckpt['current_training_step']

    def configure_optimizers(self):
        gen_params = [
            {"params": self.generator.parameters()},
            # {"params": self.contrasive_loss.parameters()},  # contrasive_loss 中的参数也需要训练
        ]
        disc_params = [
            {"params": self.multiperioddisc.parameters()},
            {"params": self.multiresdisc.parameters()},
            {"params": self.dac.parameters()},
        ]

        opt_gen = torch.optim.AdamW(gen_params, **self.config['opt_gen'])
        opt_disc = torch.optim.AdamW(disc_params, **self.config['opt_disc'])

        sch_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.config['sch']['warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )
        sch_disc = transformers.get_cosine_schedule_with_warmup(
            opt_disc, num_warmup_steps=self.config['sch']['warmup_steps'], num_training_steps=self.trainer.max_steps // 2,
        )

        return [opt_gen, opt_disc], [sch_gen, sch_disc]
    

    def calculate_adaptive_weight(self, recon_loss, adv_loss, last_layer):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        adv_grads = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(adv_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1.0).detach()
        return d_weight
    
    def get_last_layer(self,):
        return self.generator.decoder.head.out.weight
    

    # reference: SCALING TRANSFORMERS FOR LOW-BITRATE HIGHQUALITY SPEECH CODING
    def cal_perceptual_loss(self, recon, wav):
        # b,t
        with torch.no_grad():
            gt_perceptual = self.feature_extractor(wav, output_hidden_states=True).hidden_states
        gen_perceptual = self.feature_extractor(recon, output_hidden_states=True).hidden_states

        gt_perceptual_se = torch.stack(gt_perceptual, dim=1)  # b,l,t,d
        gen_perceptual_se = torch.stack(gen_perceptual, dim=1)  # b,l,t,d
        
        scale = 1 / (gt_perceptual_se.abs().mean([-2, -1]) + 1e-5)  # b,l
        perceptual_loss = (gt_perceptual_se - gen_perceptual_se).abs().mean([-2, -1]) * scale
        perceptual_loss = perceptual_loss.mean()

        return perceptual_loss

