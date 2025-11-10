echo "[DEBUG ARGS] $@"
MASTER_PORT=29501
MASTER_ADDR=localhost
NODE_RANK=0
NUM_NODES=1
BATCH_SIZE=4
SAMPLES_PER_EPOCH=1200000
DEVICES=8
save_ckpt_step=1000
CONFIG=""
COSY_YAML=""
DATA_LIST=""
DEBUG_MODE=False
WAVLM_DIR=""
CKPT=""
TIMEOUT=2.0
echo "[DEBUG ARGS] $@"
while [[ $# -gt 0 ]]; do
key="$1"
case $key in
    --batch_size)
        BATCH_SIZE="$2"; shift; shift ;;
    --devices)
        DEVICES="$2"; shift; shift ;;
    --save_ckpt_step)
        save_ckpt_step="$2"; shift; shift ;;
    --node_rank)
        NODE_RANK="$2"; shift; shift ;;
    --num_nodes)
        NUM_NODES="$2"; shift; shift ;;
    --master_addr)
        MASTER_ADDR="$2"; shift; shift ;;
    --master_port)
        MASTER_PORT="$2"; shift; shift ;;
    --samples_per_epoch)
        SAMPLES_PER_EPOCH="$2"; shift; shift ;;
    --wavlm_dir)
        WAVLM_DIR="$2"; shift; shift ;;
    --config)
        CONFIG="$2"; shift; shift ;;
    --cosy_yaml)
        COSY_YAML="$2"; shift; shift ;;
    --uio_train_data)
        DATA_LIST="$2"; shift; shift ;;
    --DEBUG_MODE)
        DEBUG_MODE="$2"; shift; shift ;;
    --ckpt)
        CKPT="$2"; shift; shift ;;
    *)

        echo "❌ Unknown arg: $1"; exit 1 ;;
esac
done


echo "MASTER_ADDR     = $MASTER_ADDR"
echo "MASTER_PORT     = $MASTER_PORT"
echo "NUM_NODES       = $NUM_NODES"
echo "NODE_RANK       = $NODE_RANK"
echo "BATCH_SIZE      = $BATCH_SIZE"
echo "DEVICES         = $DEVICES"
echo "save_ckpt_step= $save_ckpt_step"
echo "CONFIG          = $CONFIG"
echo "COSY_YAML       = $COSY_YAML"
echo "DATA_LIST       = $DATA_LIST"
echo "DEBUG_MODE       = $DEBUG_MODE"
echo "WAVLM_DIR       = $WAVLM_DIR"
echo sample per epoch: $SAMPLES_PER_EPOCH
echo "=============================="

export HF_ENDPOINT=https://hf-mirror.com



export DEBUG_MODE=$DEBUG_MODE
export WAVLM_DIR=$WAVLM_DIR
python -m torch.distributed.run \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$DEVICES \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /root/code/lscodec/train_lscodec.py \
  --batch_size $BATCH_SIZE \
  --config $CONFIG \
  --cosy_yaml $COSY_YAML \
  --uio_train_data $DATA_LIST \
  --devices $DEVICES \
  --save_ckpt_step $save_ckpt_step \
  --samples_per_epoch $SAMPLES_PER_EPOCH \
  --num_nodes $NUM_NODES



