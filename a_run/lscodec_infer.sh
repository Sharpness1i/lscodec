export HF_ENDPOINT=https://hf-mirror.com
export WAVLM_DIR=$1

cd /root/code/lscodec
export DEBUG_MODE=$2
export PYTHONPATH=/root/code/lscodec:$PYTHONPATH

ckpt=$3

python /root/code/lscodec/test_lscodec.py --config /primus_biz_workspace/zhangboyang.zby/lscodec/conf/config.yaml --save_enhanced $ckpt --recon_dir $4



# bash /root/code/lscodec/a_run/lscodec_infer.sh /mnt/wavlm_large True /root/code/lscodec/log/log1030/ckpts/version_0/step-step=100.ckpt