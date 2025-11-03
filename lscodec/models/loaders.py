import torch
import os
from typing import Union, Tuple, List
from pathlib import Path

from .compression import lscodecModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer


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


def get_lscodec(filename, device, num_codebooks, config):
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
    decoder_transformer = transformer.ProjectedTransformer(device=device, **_transformer_kwargs)
    quantizer = SplitResidualVectorQuantizer(**_quantizer_kwargs)

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

    if filename is None or not os.path.exists(filename):
        print(f"未提供权重文件或路径不存在: {filename}")
        return model

    print(f"🔍 加载模型权重: {filename}")
    allowed_prefixes = [
        "encoder",
        "decoder",
        "encoder_transformer",
        "decoder_transformer",
        "quantizer",
    ]

    # === 权重加载逻辑 ===
    if _is_safetensors(filename):
        from safetensors.torch import load_file
        state_dict = load_file(filename, device=device)
    else:
        pkg = torch.load(filename, map_location=device)
        if isinstance(pkg, dict):
            state_dict = pkg.get("state_dict") or pkg.get("model") or pkg
        else:
            state_dict = pkg

    # === 权重过滤 ===
    filtered_state_dict = {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in allowed_prefixes)}
    dropped = [k for k in state_dict.keys() if k not in filtered_state_dict]
    print(f"仅加载核心模块参数 ({len(filtered_state_dict)}/{len(state_dict)})")
    if len(dropped) > 0:
        print(f"已忽略非核心权重（示例前5项）: {dropped[:5]}")

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if missing:
        print(f"缺失参数 {len(missing)} 个 (示例): {missing[:3]}")
    if unexpected:
        print(f"未使用参数 {len(unexpected)} 个 (示例): {unexpected[:3]}")

    print("🎯 模型加载完成 (仅核心模块)")
    return model