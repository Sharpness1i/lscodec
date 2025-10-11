export HF_ENDPOINT=https://hf-mirror.com

HOST_NODE_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
num_nodes=${NNODES}
nproc_per_node=${NUM_ACCELERATORS}
node_rank=${RANK}
export NODE_RANK=$node_rank

echo "num_nodes is $num_nodes, node_rank is $node_rank, proc_per_node is $nproc_per_node"

torchrun --nproc_per_node=$NUM_ACCELERATORS --nnodes=$NNODES --node_rank=$NODE_RANK --rdzv_endpoint=${HOST_NODE_ADDR} train.py
