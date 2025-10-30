import torch
from audio_to_mel import Audio2Mel

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):

    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')

    l_t = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_f = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

    #time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav) 

    #frequency domain loss, window length is 2^i, hop length is 2^i/4, i \in [5,11]. combine l1 and l2 loss
    # 注意，这是一个multiscale mel loss
    for i in range(5, 12): # e=5,...,11
        fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=64, sampling_rate=sample_rate)
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))

    
    for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
            
            l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

    KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
    l_feat /= KL_scale
    K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
    l_g /= K_scale

    return {
        'l_t': l_t,
        'l_f': l_f,
        'l_g': l_g,
        'l_feat': l_feat,
    }

def disc_loss(logits_real, logits_fake):

    relu = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd
