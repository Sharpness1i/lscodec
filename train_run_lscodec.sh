ROOT_DIR=/root/code/lscodec
cd $ROOT_DIR
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR="192.168.1.101"
export MASTER_PORT=12355
export WORLD_SIZE=8
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH

export HUBERT_DIR=/root/code/lscodec/bosonai_hubert_base
export WAVLM_DIR=/mnt/wavlm_large
export DEBUG_MODE=False

torchrun --nproc_per_node=2 --nnodes=1 train_lscodec.py --config /root/code/lscodec/conf/config.yaml

