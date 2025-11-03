#!/bin/bash
set -e
set -o pipefail
set -x

# rdma ready
# ude: hub.docker.alibaba-inc.com/sm-primus/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04
# lingjun: dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/lingjun-pytorch-training:2.4-24.07-gu8-gpu-py310-cu125-ubuntu22.04_accelerated

MIRROR_URL="https://mirrors.cloud.aliyuncs.com/pypi/simple"
TRUSTED_HOST="mirrors.cloud.aliyuncs.com"

sudo sed -i 's#mirrors.aliyun.com#mirrors.cloud.aliyuncs.com#g' /etc/apt/sources.list
sudo apt-get update && sudo apt install -y libsox-dev ffmpeg || true #

unset PIP_EXTRA_INDEX_URL || true

for scope in global user site; do
    pip config unset "$scope.extra-index-url" 2>/dev/null || true
done

for cfg in /etc/xdg/pip/pip.conf \
           /etc/pip.conf \
           /root/.pip/pip.conf \
           /root/.config/pip/pip.conf \
           /usr/pip.conf; do
    [ -f "$cfg" ] && sed -i '/extra-index-url/d' "$cfg"
done

pip config set global.index-url "$MIRROR_URL"
pip config set global.trusted-host "$TRUSTED_HOST"

for cfg in /usr/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf; do
    [ -f "$cfg" ] && sed -i "s#pypi\.ngc\.nvidia\.com#$TRUSTED_HOST#g" "$cfg"
done

pip config list

pip install --no-cache-dir fastparquet kaldiio kaldi_native_fbank omegaconf langid nvitop
pip install --no-cache-dir accelerate==1.3.0

pip install /primus_biz_workspace/zhubingqing.zbq/workspace/torchaudio-2.0.2+31de77d-cp310-cp310-linux_x86_64.whl --no-deps

pip uninstall -y transformer_engine
until pip install --no-build-isolation "transformer_engine[pytorch]"; do
    echo "Install transformer_engine failed, retrying..."
done
echo "transformer_engine installed successfully."

pip install triton==3.1.0
pip install qwen-omni-utils

# ms-swift for dataset
# branch bqzhu/megatron
(
    cd ${PRIMUS_GIT_REPO_DIR}/ms-swift && pip install -e .
)

# transformers for tokenizer
# branch bqzhu/dev
(
    cd ${PRIMUS_GIT_REPO_DIR}/transformers && pip install -e .
)

pip install -r cosy_requirements.txt

pip install deepspeed==0.15.4
pip install -U openai-whisper