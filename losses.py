import torch
from audio_to_mel import Audio2Mel




def multiscale_mel_loss(input_wav, output_wav, sample_rate=24000):
    
    device = input_wav.device
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')

    l_f = torch.tensor(0.0, device=device)
    for i in range(5, 12):
        fft = Audio2Mel(
            n_fft=2**i, win_length=2**i, hop_length=(2**i)//4,
            n_mel_channels=64, sampling_rate=sample_rate
        )
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))
    return l_f

def waveform_loss(input_wav, output_wav):
    device = input_wav.device
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l_t = l1Loss(input_wav, output_wav)
    return l_t






def total_loss(
    fmap_real, logits_fake, fmap_fake,
    input_wav, output_wav, sample_rate=24000,
    use_gan=True, use_fm=True,
):
    device = input_wav.device
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')

    # time / freq losses 一直可用
    l_t = l1Loss(input_wav, output_wav)

    l_f = torch.tensor(0.0, device=device)
    for i in range(5, 12):
        fft = Audio2Mel(
            n_fft=2**i, win_length=2**i, hop_length=(2**i)//4,
            n_mel_channels=64, sampling_rate=sample_rate
        )
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))

    # adversarial 与 feature matching 可关
    if use_gan and (logits_fake is not None):
        l_g = torch.tensor(0.0, device=device)
        K_scale = len(logits_fake)
        for tt1 in range(K_scale):
            l_g = l_g + torch.mean(relu(1 - logits_fake[tt1]))
        l_g = l_g / K_scale
    else:
        l_g = torch.tensor(0.0, device=device)

    if use_fm and (fmap_real is not None) and (fmap_fake is not None):
        l_feat = torch.tensor(0.0, device=device)
        KL_scale = len(fmap_real) * len(fmap_real[0])
        for tt1 in range(len(fmap_real)):
            for tt2 in range(len(fmap_real[tt1])):
                # 避免除零
                denom = torch.mean(torch.abs(fmap_real[tt1][tt2])) + 1e-8
                l_feat = l_feat + torch.nn.functional.l1_loss(
                    fmap_real[tt1][tt2], fmap_fake[tt1][tt2], reduction='mean'
                ) / denom
        l_feat = l_feat / KL_scale
    else:
        l_feat = torch.tensor(0.0, device=device)

    return {'l_t': l_t, 'l_f': l_f, 'l_g': l_g, 'l_feat': l_feat}

def disc_loss(logits_real, logits_fake):

    relu = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd
