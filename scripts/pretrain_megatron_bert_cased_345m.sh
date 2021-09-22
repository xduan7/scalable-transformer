#!/usr/bin/env bash

# Pre-train Megatron-LM BERT (cased with 345m parameters) on single
#   machine node (with one or more GPUs)
#
# Arguments:
#   -d or --distributed: option for multi-GPU training
#
# Usage:
#   $ ./pretrain_megatron_bert_cased_345m.sh  # single GPU
#   or
#   $ ./pretrain_megatron_bert_cased_345m.sh -d  # multiple GPUs on a single node
#
# Note:
#   This script is only good for debugging on single-node.
#   Please use `./submit_pretraining_job.sh` for training job
#   submission with Cobalt on single node or multiple nodes with MPI.

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

# LOAD_CHECKPOINT_PATH="${MEGATRON_BERT_CASED_345M_CHECKPOINT_DIR_PATH}"
SAVE_CHECKPOINT_PATH="${RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH}"
VOCAB_FILE="${BERT_CASED_ENWIKI_VOCAB_PATH}"
DATA_PATH="${BERT_CASED_ENWIKI_TEXT_DIR_PATH}/text_sentence"

BERT_ARGS="\
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --lr 0.0001 \
  --lr-decay-iters 50 \
  --train-iters 200 \
  --eval-interval 2000 \
  --min-lr 0.00001 \
  --lr-warmup-fraction 0.01 \
  --vocab-file ${VOCAB_FILE} \
  --split 949,50,1 \
  --fp16 \
"

OUTPUT_ARGS="\
  --log-interval 10 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations \
"

if  [ "${1}" = "-d" ] || [ "${1}" = "--distributed" ]; then
  # Get the following variables from `distributed_params.sh`:
  # - DISTRIBUTED_MODULE
  # - DISTRIBUTED_MODULE_ARGS
  # - DISTRIBUTED_ARGS
  # - BERT_BATCH_ARGS
  source "${CURR_DIR_PATH}/distributed_params.sh"
else
  MICRO_BATCH_SIZE=4
  GLOBAL_BATCH_SIZE=8
fi
BERT_BATCH_ARGS="\
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
"

# Some parallelized models are not compatible with the OG Megatron-LM checkpoint
# Add `--load ${LOAD_CHECKPOINT_PATH}` into the following command for loading
#   the OG Megatron-LM checkpoints,
#   --save ${SAVE_CHECKPOINT_PATH} \
PYTHON_CMD="${MEGATRON_DS_CONTAINER_PYTHON_PATH} \
  ${DISTRIBUTED_MODULE} ${DISTRIBUTED_MODULE_ARGS} \
  ${MEGATRON_DIR_PATH}/pretrain_bert.py \
  ${BERT_ARGS} \
  ${BERT_BATCH_ARGS} \
  ${DISTRIBUTED_ARGS} \
  ${OUTPUT_ARGS} \
  --data-path ${DATA_PATH} \
"

# shellcheck disable=SC2086
singularity exec \
  --nv \
  --cleanenv \
  --bind /gpfs,/lus \
  ${MEGATRON_DS_CONTAINER_PATH} \
  ${PYTHON_CMD}