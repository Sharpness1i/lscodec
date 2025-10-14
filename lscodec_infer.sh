
## 

export HF_ENDPOINT=https://hf-mirror.com
export WAVLM_DIR=$1

cd /root/code/lscodec
export DEBUG_MODE=$2
export PYTHONPATH=/root/code/lscodec:$PYTHONPATH

ckpt=$3

python /root/code/lscodec/test_lscodec.py --config /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml --save_enhanced $ckpt



# bash /root/code/lscodec/lscodec_infer.sh /primus_biz_workspace/zhangboyang.zby/CKPT/wavlm_large False /root/code/lscodec/log/log1001/ckpts/version_7/epoch=4-last.ckpt