import typing
from typing import List
import torch
import torch.nn as nn
import torchaudio
import torchaudio
from einops import rearrange


def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))



class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self, sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 100,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=True, power=1,
        )
    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the mel-scaled magnitude spectrograms.
        """
        mel_hat = safe_log(self.mel_spec(y_hat))
        mel = safe_log(self.mel_spec(y))

        loss = torch.nn.functional.l1_loss(mel, mel_hat)
        return loss

class STFTSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the magnitude spectrograms of the ground truth sample and the generated sample
    """

    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
    ):
        super().__init__()
        self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=True, power=1)

    def forward(self, y_hat, y) -> torch.Tensor:
        """
        Args:
            y_hat (Tensor): Predicted audio waveform.
            y (Tensor): Ground truth audio waveform.

        Returns:
            Tensor: L1 loss between the magnitude spectrograms.
        """
        mag_hat = safe_log(self.transform(y_hat))
        mag = safe_log(self.transform(y))

        loss = torch.nn.functional.l1_loss(mag, mag_hat)
        return loss







