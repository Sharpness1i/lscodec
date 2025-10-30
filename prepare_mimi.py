
#'/mnt/huggingface/hub/models--kyutai--lscodecko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'


from huggingface_hub import hf_hub_download
import torch
from lscodec.models import loaders, LMGen
import debugpy; debugpy.listen(('0.0.0.0', 5678)); print('I am waiting for you');debugpy.wait_for_client();debugpy.breakpoint();

lscodec_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.lscodec_NAME)
lscodec = loaders.get_lscodec(lscodec_weight, device='cpu')
lscodec.set_num_codebooks(8) 