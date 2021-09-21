#!/usr/bin/env bash

# Pre-train Megatron-LM BERT (cased with 175b parameters) on single
#   machine node (with one or more GPUs)
#
# Usage:
#   $ ./pretrain_megatron_bert_cased_175b.sh
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

BERT_ARGS="\
  --num-layers 48 \
  --hidden-size 12288 \
  --num-attention-heads 96 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --train-samples 146484375 \
  --lr-decay-samples 126953125 \
  --lr-warmup-samples 183105 \
  --lr 6.0e-5 \
	--min-lr 6.0e-6 \
	--lr-decay-style cosine \
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

# Get the following variables from `distributed_params.sh`:
# - DISTRIBUTED_MODULE
# - DISTRIBUTED_MODULE_ARGS
# - DISTRIBUTED_ARGS
# - BERT_BATCH_ARGS
source "${CURR_DIR_PATH}/distributed_params.sh"
BERT_BATCH_ARGS="\
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
"

# Some parallelized models are not compatible with the OG Megatron-LM checkpoint
# Add `--load ${LOAD_CHECKPOINT_PATH}` into the following command for loading
#   the OG Megatron-LM checkpoints,
PYTHON_CMD="${CONTAINER_PYTHON_PATH} \
  ${DISTRIBUTED_MODULE} ${DISTRIBUTED_MODULE_ARGS} \
  ${MEGATRON_DIR_PATH}/pretrain_bert.py \
  ${BERT_ARGS} \
  ${BERT_BATCH_ARGS} \
  ${DISTRIBUTED_ARGS} \
  ${OUTPUT_ARGS} \
  --save ${SAVE_CHECKPOINT_PATH} \
  --data-path ${DATA_PATH} \
"

# shellcheck disable=SC2086
singularity exec \
  --nv \
  --cleanenv \
  --bind /gpfs,/lus \
  ${CONTAINER_PATH} \
  ${PYTHON_CMD}