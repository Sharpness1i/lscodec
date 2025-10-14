
## 

export HF_ENDPOINT=https://hf-mirror.com
export WAVLM_DIR=$1

cd /root/code/lscodec
export DEBUG_MODE=$2
export PYTHONPATH=/root/code/lscodec:$PYTHONPATH
python /root/code/lscodec/test_lscodec.py --config /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml 



# bash /root/code/lscodec/lscodec_infer.sh /mnt/wavlm_large True