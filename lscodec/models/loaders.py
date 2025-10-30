from pathlib import Path
from .compression import lscodecModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer
import torch

import torch
import os
from typing import Union, Tuple, List

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 16,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
    "q_dropout": True,
    "no_quantization_rate": 0.0
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}



def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")



import torch
import os
from typing import Union, Tuple, List

def load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[str], List[str]]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"权重文件不存在: {filename}")

  
    if str(filename).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(filename, device=device)
        except ImportError:
            raise RuntimeError("需要安装 safetensors: pip install safetensors")
    else:
        state_dict = torch.load(filename, map_location=device)


    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)


    def long_prefixes(keys):
        prefixes = set()
        for k in keys:
            parts = k.split(".")
            prefixes.add(".".join(parts[:5]) if len(parts) >= 2 else parts[0])
        return sorted(prefixes)

    def short_prefixes(keys):
        prefixes = set()
        for k in keys:
            parts = k.split(".")
            prefixes.add(".".join(parts[:2]) if len(parts) >= 2 else parts[0])
        return sorted(prefixes)
    
    short_missing = short_prefixes(missing_keys)
    short_unexpected = long_prefixes(unexpected_keys)

    if strict:
        if missing_keys or unexpected_keys:
            msg = f"加载权重时存在不匹配:\n"
            if short_missing:
                msg += f"  缺失权重模块: {short_missing}\n"
            if short_unexpected:
                msg += f"  多余权重模块: {short_unexpected}\n"
            raise RuntimeError(msg)
    else:
        if short_missing:
            print(f"忽略缺失权重模块: {short_missing}")
        if short_unexpected:
            print(f"忽略多余权重模块: {short_unexpected}")

    print(f"模型权重加载完成: {filename}  (strict={strict})")
    return missing_keys, unexpected_keys


def get_lscodec(
    filename, device, num_codebooks,config
) -> lscodecModel:
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = lscodecModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
        config=config,
    ).to(device=device)

    model.set_num_codebooks(num_codebooks)
    
    if filename is not None:
        if _is_safetensors(filename):
            load_model(model, filename, device=str(device),strict=False)
        else:
            pkg = torch.load(filename, "cpu")
            model.load_state_dict(pkg["model"])
    
    return model




