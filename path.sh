source /root/miniconda3/bin/activate /root/miniconda3/envs/unified_llm_audio

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

MAX_THREADS=3
export MKL_NUM_THREADS=$MAX_THREADS
export NUMEXPR_NUM_THREADS=$MAX_THREADS
export OMP_NUM_THREADS=$MAX_THREADS
export OPENBLAS_NUM_THREADS=$MAX_THREADS
export VECLIB_MAXIMUM_THREADS=$MAX_THREADS

export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_PLUGIN=none
# for dataloader ddp
export GLOO_SOCKET_IFNAME=eth0

bash ./torchrun.sh