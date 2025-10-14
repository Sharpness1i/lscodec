ROOT_DIR=/root/code/lscodec
find $ROOT_DIR -name '*.pyc' -delete
export HF_ENDPOINT=https://hf-mirror.com

sudo sed -i 's#http://archive.ubuntu.com/#https://mirrors.aliyun.com/#' /etc/apt/sources.list
sudo apt-get update

(
    wget "https://eflops-network.oss-cn-beijing.aliyuncs.com/ubuntu/nic-lib-rdma-core-installer-ubuntu.tar.gz" && \
    tar -zxvf nic-lib-rdma-core-installer-ubuntu.tar.gz && \
    cd nic-lib-rdma-core-installer-ubuntu/ && bash ./install.sh && bash ./check.sh
)

source /root/miniconda3/bin/activate /root/miniconda3/envs/unified_llm_audio

export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH


pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && pip config set install.trusted-host repo.huaweicloud.com



pip install fastparquet kaldiio kaldi_native_fbank onnx qwen_omni_utils onnxruntime hyperpyyaml langid
pip install einops==0.7.0 safetensors==0.4.4 sentencepiece==0.2.0 sounddevice==0.5.0 soundfile==0.12.1 sphn==0.1.4 numpy==1.26.4 "aiohttp>=3.10.5,<3.11" huggingface-hub==0.34.4 pytest==8.3.3

pip install natsort


pip install -r requirements.txt



