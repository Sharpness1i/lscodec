ROOT_DIR=/root/code/lscodec


GPU_NUM=$1

cd $ROOT_DIR
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR="192.168.1.101"
export MASTER_PORT=12355
export WORLD_SIZE=8
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/src:$PYTHONPATH

export HUBERT_DIR=/root/code/lscodec/bosonai_hubert_base

export WAVLM_DIR=$2

export DEBUG_MODE=$3

torchrun --nproc_per_node=$GPU_NUM --nnodes=1 train_lscodec.py --config /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml


# bash /root/code/lscodec/train_run_lscodec.sh 1 /mnt/wavlm_large True

