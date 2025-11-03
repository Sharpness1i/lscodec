
cd /root/code/lscodec
export HF_ENDPOINT=https://hf-mirror.com
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT=12355
export WORLD_SIZE=8
echo "ℹ️  当前环境变量:"
echo "     NUM_ACCELERATORS = $NUM_ACCELERATORS"
echo "     NNODES = $NNODES"
#source /root/miniconda3/bin/activate /root/miniconda3/envs/unified_llm_audio
#bash /primus_biz_workspace/yanhaoyin.yhy/init.sh #init部分整合补丁

torchrun --nproc_per_node=$NUM_ACCELERATORS --nnodes=$NNODES train.py


