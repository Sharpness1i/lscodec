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

config_file=$4

data_list=$5

torchrun --nproc_per_node=$GPU_NUM --nnodes=1 train_lscodec.py --config $config_file --cosy_yaml /root/code/lscodec/cosy_conf/cosyvoice2_ori.yaml --uio_train_data $data_list

    
# bash /root/code/lscodec/a_run/train_run_lscodec.sh 1 /mnt/wavlm_large True /root/code/lscodec/conf/config.yaml /primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list

# bash /root/code/lscodec/a_run/train_run_lscodec.sh 2 /mnt/wavlm_large False /root/code/lscodec/conf/config.yaml


# bash /root/code/lscodec/a_run/train_run_lscodec.sh 1 /mnt/wavlm_large True /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml
