ENVIR=${ENVIR:-1}
if [[ $ENVIR -eq 1 ]]; then
    echo "🔹 执行 prepare_env.sh ..."
    bash ./prepare_env.sh || exit 1
    . ./path.sh 
else
    echo "⚠️ 跳过 prepare_env.sh"
    echo "⚠️ 跳过 . .path.sh"
fi

node_rank=${RANK}
PRIMUS_OUTPUT_DIR=${PRIMUS_OUTPUT_DIR:-$(pwd)}
HOST_NODE_ADDR="${MASTER_ADDR}:${MASTER_PORT}"
num_nodes=${NNODES}
nproc_per_node=${NUM_ACCELERATORS}

use_uio_dataset=
job_id="zhubingqing.zbq"
rdzv_backend="static"
gpu_list=
use_parquet_dataset_conf=
train_type=
# UIO dataset
uio_dataset_conf=
dataloader_num_workers=8
uio_pin_memory=true
uio_prefetch=500
train_set=train
dev_set=dev
# For random_select_tar
train_data=
random_select_tar_numbers=

batch_size=16
save_steps=10000
eval_steps=100000000
num_train_epochs=1
target_modules=
model_type="qwen2_audio"
model_id_or_path=
resume_from_checkpoint=
resume_only_model=
check_model=
load_dataset_config=
ignore_data_skip=

sft_type="full"
sft_cmd="sft"
learning_rate=

max_steps=
max_length=

deepspeed=

gradient_accumulation_steps=1
gradient_checkpointing=true
gradient_checkpointing_kwargs=

freeze_parameters_ratio=
trainable_parameters=

train_sub_module=
freeze_talker_llm=
freeze_talker_mtp=
system=
template=
template_backend=
freeze_llm=

# LoRA
lora_rank=
lora_alpha=
modules_to_save=
freeze_vit=
freeze_aligner=

. utils/parse_options.sh || exit 1;

[ ! -z $uio_dataset_conf ] || exit 1

data_dir=$1
exp_dir=$2

set -e
set -u
set -o pipefail
set -x

[ ! -d $exp_dir ] && mkdir -p $exp_dir

[ $sft_cmd == "pt" ] && sft_type="full"

echo "RANK${RANK} " > ${PRIMUS_OUTPUT_DIR}/common_ip_${RANK}.txt
counter=`cat ${PRIMUS_OUTPUT_DIR}/common_ip_*.txt | wc -l `
while [ $counter -lt ${NNODES} ]
do
    echo "Wait for all nodes to be ready, current counter: ${counter}, all node: ${NNODES}"
    sleep 5
    counter=`cat ${PRIMUS_OUTPUT_DIR}/common_ip_*.txt | wc -l`
done
sleep 2
rm -f ${PRIMUS_OUTPUT_DIR}/common_ip_*.txt

echo "$0: num_nodes is $num_nodes, node_rank is $node_rank, proc_per_node is $nproc_per_node"

export NODE_RANK=$node_rank

NPROC_PER_NODE=$nproc_per_node \
NNODES=$num_nodes \
NODE_RANK=$node_rank \
RDZV_BACKEND=$rdzv_backend \
RDZV_ENDPOINT=$HOST_NODE_ADDR \
RDZV_ID=$job_id \
swift $sft_cmd \
    --model_type $model_type \
    ${model_id_or_path:+--model $model_id_or_path} \
    --use_uio_dataset $use_uio_dataset \
    --uio_dataset_conf $uio_dataset_conf \
    --use_parquet_dataset_conf $use_parquet_dataset_conf \
    --uio_train_data "${train_data:-$data_dir/$train_set/data.list}" \
    ${random_select_tar_numbers:+--uio_random_select_tar_numbers "$random_select_tar_numbers"} \
    ${dataloader_num_workers:+--dataloader_num_workers $dataloader_num_workers} \
    ${train_type:+--train_type $train_type} \
    ${target_modules:+--target_modules $target_modules} \
    --train_type $sft_type \
    --per_device_train_batch $batch_size \
    --gradient_checkpointing $gradient_checkpointing \
    ${gradient_checkpointing_kwargs:+--gradient_checkpointing_kwargs "$gradient_checkpointing_kwargs"} \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --output_dir $exp_dir \
    --save_steps $save_steps \
    --eval_steps $eval_steps \
    --ddp_backend nccl \
    --num_train_epochs $num_train_epochs \
    ${deepspeed:+--deepspeed $deepspeed} \
    ${train_sub_module:+--train_sub_module $train_sub_module} \
    ${freeze_talker_llm:+--freeze_talker_llm $freeze_talker_llm} \
    ${freeze_talker_mtp:+--freeze_talker_mtp $freeze_talker_mtp} \
    ${freeze_llm:+--freeze_llm $freeze_llm} \
    ${freeze_parameters_ratio:+--freeze_parameters_ratio $freeze_parameters_ratio} \
    ${trainable_parameters:+--trainable_parameters $trainable_parameters} \
    ${lora_rank:+--lora_rank $lora_rank} \
    ${lora_alpha:+--lora_alpha $lora_alpha} \
    ${modules_to_save:+--modules_to_save $modules_to_save} \
    ${freeze_vit:+--freeze_vit $freeze_vit} \
    ${freeze_aligner:+--freeze_aligner $freeze_aligner} \
    ${learning_rate:+--learning_rate $learning_rate} \
    ${resume_from_checkpoint:+--resume_from_checkpoint $resume_from_checkpoint} \
    ${resume_only_model:+--resume_only_model $resume_only_model} \
    ${check_model:+--check_model $check_model} \
    ${load_dataset_config:+--load_dataset_config $load_dataset_config} \
    ${ignore_data_skip:+--ignore_data_skip $ignore_data_skip} \
    ${max_steps:+--max_steps $max_steps} \
    ${max_length:+--max_length $max_length} \
    ${system+--system} "${system}" \
    ${template:+--template $template} \
    ${template_backend:+--template_backend $template_backend}


