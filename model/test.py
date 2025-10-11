import torch
from vq import Codec


encoder_kwargs = {
    "channels": 1,
    "dimension": 512,
    "n_filters": 32,
    "ratios": [2,4,5,8],
    "causal": False,
}

decoder_kwargs = {
    "input_channels": 512,
    "dim": 768,
    "intermediate_dim": 768*3,
    "convnext_layers": 12,
    "n_fft": 1280,
    "hop_length": 320,  # np.prod(ratios)
    "causal": False,
}

# quantizer_kwargs = {
#     "n_e": 8192,
#     "e_dim": 512,
# }
quantizer_kwargs = {
    "dim": 512,
    "codebook_size": 4096,
    "num_quantizers": 1,
    "decay": 0.99,
    "kmeans_init": True,
    "kmeans_iters": 200,
    "threshold_ema_dead_code": 2,
}

codec = Codec(encoder_kwargs, decoder_kwargs, quantizer_kwargs).cuda()


x = torch.randn(2, int(4.5 * 16000)).cuda()
feat = codec.extract_wav2vec2_features(x)

import ipdb; ipdb.set_trace()
print(feat.shape)

# from ptflops import get_model_complexity_info
# flops, params = get_model_complexity_info(codec, (1, 16000), as_strings=True,
#                                             print_per_layer_stat=True, verbose=True)
# print(flops, params)

# print(sum([p.numel() for p in codec.encoder.parameters()]))
# print(sum([p.numel() for p in codec.decoder.parameters()]))

# x = torch.randn(2, 1, 16000).cuda()

# recon, commit_loss, cnn_feat, mask_indices, quantized = codec(x, use_mask=True, domain_split=[0.5, 1.0])

# # import ipdb; ipdb.set_trace()

# print(recon.shape)  # [B, T]

