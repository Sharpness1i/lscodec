set -e
NNODES=2            
GPUS_PER_NODE=8     
MASTER_ADDR=$1
MASTER_PORT=$2
NODE_RANK=$3

CONFIG=$4
COSY_YAML=$5
TRAIN_LIST=$6

echo "-----------------------------------------"
echo "Launching Distributed Training"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"

echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "-----------------------------------------"


export HF_ENDPOINT=https://hf-mirror.com && export WAVLM_DIR=/mnt/wavlm_large && export PYTHONPATH=/root/code/lscodec:$PYTHONPATH
python -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /root/code/lscodec/train_lscodec.py \
  --config $CONFIG \
  --cosy_yaml $COSY_YAML \
  --uio_train_data $TRAIN_LIST


  