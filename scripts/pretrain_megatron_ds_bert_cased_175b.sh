#!/usr/bin/env bash

# Pre-train Megatron-LM DeepSpeed BERT (cased with 175b parameters) on single
#   machine node (with one or more GPUs)
#
# Arguments:
#   -d or --distributed: option for multi-GPU training
#
# Usage:
#   $ ./pretrain_megatron_bert_cased_175b.sh  # single GPU
#   or
#   $ ./pretrain_megatron_bert_cased_175b.sh -d  # multiple GPUs
#
# Note:
#   This script is only good for debugging on single-node.
#   Please use `./submit_pretraining_job.sh` for training job
#   submission with Cobalt on single node or multiple nodes with MPI.

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

SAVE_CHECKPOINT_PATH="${RETRAINED_BERT_CASED_175B_CHECKPOINT_DIR_PATH}"
VOCAB_FILE="${BERT_CASED_ENWIKI_VOCAB_PATH}"
DATA_PATH="${BERT_CASED_ENWIKI_TEXT_DIR_PATH}/text_sentence"
DS_CONFIG_PATH="${DS_ZERO3_MINIMAL_CONFIG_PATH}"


BERT_ARGS="\
  --num-layers 96 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --lr 6.0e-5 \
  --lr-decay-iters 990000 \
  --train-iters 2000000 \
	--min-lr 6.0e-6 \
	--lr-decay-style cosine \
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
  MICRO_BATCH_SIZE=1
else
  NODE_RANK=0
  MICRO_BATCH_SIZE=4
fi
BERT_BATCH_ARGS="\
  --batch-size ${MICRO_BATCH_SIZE} \
"

PYTHON_CMD="${MEGATRON_DS_CONTAINER_PYTHON_PATH} \
  ${DISTRIBUTED_MODULE} ${DISTRIBUTED_MODULE_ARGS} \
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
singularity exec \
  --nv \
  --cleanenv \
  --env LOCAL_RANK="${NODE_RANK}" \
  --bind /gpfs,/lus \
  ${MEGATRON_DS_CONTAINER_PATH} \
  ${PYTHON_CMD}
