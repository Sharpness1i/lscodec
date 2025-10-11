import torch
from torch import nn
import torch.nn.functional as F
from .encoder_modules import SEANetEncoder as CodecEncoder
from .codec_decoder import CodecDecoder
from vector_quantize_pytorch.residual_vq import ResidualVQ
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d



class Codec(nn.Module):
    def __init__(
        self, 
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        quantizer_kwargs: dict,
    ):
        super().__init__()

        self.encoder = CodecEncoder(
            causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
            dimension=512, channels=1, n_filters=32, ratios=[8, 5, 4, 2], activation='ELU',
            kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
            true_skip=False, compress=2, use_transformer=True,
        )
        
        self.decoder = CodecDecoder(
            input_channels=512 * 2,
            dim=768,
            intermediate_dim=2304,
            # num_layers=12,
        )

        self.quantizer = ResidualVQ(
            dim = 512,
            codebook_size = 1024,
            num_quantizers = 4,
            decay = 0.99,
            kmeans_init = True,
            kmeans_iters = 50,
            quantize_dropout = True,
        )

        self.semantic_quantizer = ResidualVQ(
            dim = 512,
            codebook_size = 1024,
            num_quantizers = 4,
            decay = 0.99,
            kmeans_init = True,
            kmeans_iters = 50,
            quantize_dropout = True,
        )


        self.semantic_encoder = SemanticEncoder(
            input_channels=768,
            encode_channels=768,
            out_channels=512,
            channel_ratios=(1, 1),
            strides=(2, 1),
        )

        self.semantic_decoder = SemanticDecoder(
            code_dim=512,
            output_channels=768,
            decode_channels=768,
            channel_ratios=(1, 1),
            strides=(2, 1),
        )
    
    def forward(self, x, feat, use_mask=False, domain_split=None):

        emb = self.encoder(x) # (b,512,t)
        semantic_emb = self.semantic_encoder(feat) # (b,512,t)

        
        # 声学codec
        quantized, codes, commit_loss = self.quantizer(emb.transpose(-2, -1))
        
        quantized = quantized.transpose(-2, -1)
        commit_loss = commit_loss.mean()

        # 语义codec
        quantized_semantic, codes_semantic, commit_loss_semantic = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)
        commit_loss_semantic = commit_loss_semantic.mean()

        recon = self.decoder(torch.cat([quantized, quantized_semantic], dim=1))

        pred_feat = self.semantic_decoder(quantized_semantic)
        # recon = self.head(x)
        return recon, pred_feat, (commit_loss + commit_loss_semantic).mean()


    @torch.no_grad()
    def encode(self, x, feat, use_mask=False, domain_split=None):
        pass
    
    @torch.no_grad()
    def decode(self, codes):
        pass
    
    @torch.no_grad()
    def get_quantized_emb(self, x, feat):
        pass



