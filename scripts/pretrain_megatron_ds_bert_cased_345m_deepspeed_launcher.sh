#!/usr/bin/env bash

# Pre-train Megatron-LM DeepSpeed BERT (cased with 345m parameters) on single
#   machine node (with one or more GPUs) with deepspeed launcher
#
# Arguments:
#   -d or --distributed: option for multi-GPU training
#
# Usage:
#   $ ./pretrain_megatron_ds_bert_cased_345m_deepspeed_launcher.sh  # single GPU
#   or
#   $ ./pretrain_megatron_ds_bert_cased_345m_deepspeed_launcher.sh -d  # multiple GPUs
#
# Note:
#   This script is only good for debugging on single-node.
#   Please use `./submit_pretraining_job.sh` for training job
#   submission with Cobalt on single node or multiple nodes with MPI.

echo "Pretraining launched with DeepSpeed does not work due to ssh failure."

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

SAVE_CHECKPOINT_PATH="${RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH}"
VOCAB_FILE="${BERT_CASED_ENWIKI_VOCAB_PATH}"
DATA_PATH="${BERT_CASED_ENWIKI_TEXT_DIR_PATH}/text_sentence"
DS_CONFIG_PATH="${DS_ZERO0_OFFLOAD_MINIMAL_CONFIG_PATH}"


BERT_ARGS="\
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --lr 0.0001 \
  --lr-decay-iters 990000 \
  --train-iters 2000000 \
  --min-lr 0.00001 \
  --warmup 0.01 \
  --vocab-file ${VOCAB_FILE} \
  --split 949,50,1 \
  --fp16 \
"

MEGATRON_DS_ARGS=" \
  --deepspeed \
  --deepspeed_config ${DS_CONFIG_PATH} \
"

OUTPUT_ARGS="\
  --log-interval 10 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations \
"

if  [ "$1" = "-d" ] || [ "$1" = "--distributed" ]; then
  # Get the following variables from `distributed_params.sh`:
  # - NODE_RANK
  # - DISTRIBUTED_MODULE
  # - DISTRIBUTED_MODULE_ARGS
  # - MICRO_BATCH_SIZE
  source "${CURR_DIR_PATH}/distributed_params.sh"
  # The older version of megatron does not accept
  # `tensor-model-parallel-size` and `pipeline-model-parallel-size`
  DISTRIBUTED_ARGS="\
    --model-parallel-size ${PIPELINE_MP_SIZE} \
  "
  DEEPSPEED_MPI_ARGS="\
    --hostfile ${DS_HOSTFILE_PATH} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
  "
  MEGATRON_DS_ARGS="
    ${MEGATRON_DS_ARGS} \
    --deepspeed_mpi False \
  "
else
  NNODES=1
  NODE_RANK=0
  GPUS_PER_NODE=1
  MICRO_BATCH_SIZE=4
fi
BERT_BATCH_ARGS="\
  --batch-size ${MICRO_BATCH_SIZE} \
"

#PYTHON_CMD="${CONTAINER_PYTHON_PATH} \
#  ${DISTRIBUTED_MODULE} ${DISTRIBUTED_MODULE_ARGS} \
#  ${MEGATRON_DS_DIR_PATH}/pretrain_bert.py \
#  ${BERT_ARGS} \
#  ${BERT_BATCH_ARGS} \
#  ${MEGATRON_DS_ARGS} \
#  ${DISTRIBUTED_ARGS} \
#  ${OUTPUT_ARGS} \
#  --save ${SAVE_CHECKPOINT_PATH} \
#  --data-path ${DATA_PATH} \
#"

DEEPSPEED_CMD= \
  "${DEEPSPEED_PATH} \
  ${DEEPSPEED_MPI_ARGS} \
  --num_nodes $( cat "${DS_HOSTFILE_PATH}" | wc -l ) \
  --num_gpus ${GPUS_PER_NODE} \
  ${MEGATRON_DS_DIR_PATH}/pretrain_bert.py \
  ${BERT_ARGS} \
  ${BERT_BATCH_ARGS} \
  ${MEGATRON_DS_ARGS} \
  ${DISTRIBUTED_ARGS} \
  ${OUTPUT_ARGS} \
  --save ${SAVE_CHECKPOINT_PATH} \
  --data-path ${DATA_PATH} \
"

# shellcheck disable=SC2086
# --env PATH="/home/xduan7/.local/bin/:\${PATH}" \
# LOCAL_RANK="${NODE_RANK}"
singularity exec \
  --nv \
  --cleanenv \
  --env PATH="${PDSH_PATH}:\${PATH}" \
  --bind /gpfs,/lus,${SSH_AUTH_SOCK},"/etc/ssh/ssh_config" \
  ${CONTAINER_PATH} \
  ${DEEPSPEED_CMD}
