
# 单机单卡
bash /root/code/lscodec/a_run/train_run_lscodec.sh \
  --batch_size 4 \
  --devices 1 \
  --num_nodes 1 \
  --interval_samples 2000 \
  --config /root/code/lscodec/conf/config.yaml \
  --cosy_yaml /root/code/lscodec/cosy_conf/cosyvoice2_ori.yaml \
  --uio_train_data /primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list  \
  --wavlm_dir /mnt/wavlm_large \
  --samples_per_epoch 1200000 \
  --DEBUG_MODE true


# 单机多卡
# bash /root/code/lscodec/a_run/train_run_lscodec.sh \
#   --batch_size 32 \
#   --devices 2 \
#   --num_nodes 1 \
#   --interval_samples 2000 \
#   --config /root/code/lscodec/conf/config.yaml \
#   --cosy_yaml /root/code/lscodec/cosy_conf/cosyvoice2_ori.yaml \
#   --uio_train_data /primus_biz_workspace/zhangboyang.zby/data/emilia/train/data.list  \
#   --wavlm_dir /mnt/wavlm_large \
#    --samples_per_epoch 1200000 \
#   --DEBUG_MODE False





