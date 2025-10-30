export HF_ENDPOINT=https://hf-mirror.com
export WAVLM_DIR=$1
cd /root/code/lscodec
export DEBUG_MODE=$2
export PYTHONPATH=/root/code/lscodec:$PYTHONPATH

ckpt=$3


export HUBERT_DIR=/root/code/lscodec/bosonai_hubert_base

python /root/code/lscodec/test_hcodec.py --config /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml --save_enhanced $ckpt



# bash /root/code/lscodec/lscodec_infer.sh /mnt/wavlm_large True /root/code/lscodec/log/log1001/ckpts/version_6/epoch=90-last.ckpt