#!/usr/bin/env bash

# Pre-train Megatron-LM DeepSpeed BERT (cased with 345m parameters) on single
#   machine node (with one or more GPUs)
#
# Arguments:
#   -d or --distributed: option for multi-GPU training
#
# Usage:
#   $ ./pretrain_megatron_bert_cased_345m.sh  # single GPU
#   or
#   $ ./pretrain_megatron_bert_cased_345m.sh -d  # multiple GPUs
#
# Note:
#   This script is only good for debugging on single-node.
#   Please use `./submit_pretraining_job.sh` for training job
#   submission with Cobalt on single node or multiple nodes with MPI.

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

SAVE_CHECKPOINT_PATH="${RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH}"
VOCAB_FILE="${BERT_CASED_ENWIKI_VOCAB_PATH}"
DATA_PATH="${BERT_CASED_ENWIKI_TEXT_DIR_PATH}/text_sentence"
DS_CONFIG_PATH="${DS_ZERO3_CONFIG_PATH}"


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

DEEPSPEED_ARGS=" \
  --deepspeed \
  --deepspeed_config ${DS_CONFIG_PATH} \
  --zero-stage 3 \
  --zero-reduce-bucket-size 50000000 \
  --zero-allgather-bucket-size 5000000000 \
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
  # - NNODES
  # - GPUS_PER_NODE
  # - MICRO_BATCH_SIZE
  source "${CURR_DIR_PATH}/distributed_params.sh"
else
  NNODES=1
  GPUS_PER_NODE=1
  MICRO_BATCH_SIZE=4
fi
BERT_BATCH_ARGS="\
  --batch-size ${MICRO_BATCH_SIZE} \
"


# Optioanla: dd `--load ${LOAD_CHECKPOINT_PATH}` into the following command
PYTHON_CMD="deepspeed \
  --num_nodes ${NNODES} \
  --num_gpus ${GPUS_PER_NODE} \
  ${MEGATRON_DS_DIR_PATH}/pretrain_bert.py \
  ${BERT_ARGS} \
  ${BERT_BATCH_ARGS} \
  ${DEEPSPEED_ARGS} \
  ${OUTPUT_ARGS} \
  --save ${SAVE_CHECKPOINT_PATH} \
  --data-path ${DATA_PATH}"

# `PYTHON_CMD` must be split into globs for argparse parsing
# shellcheck disable=SC2086
singularity exec \
  --nv \
  --cleanenv \
  --env PATH="/home/xduan7/.local/bin:\${PATH}" \
  --bind /gpfs/mira-home/xduan7,/lus/theta-fs0/projects/candle_aesp/xduan7/data \
  ${CONTAINER_PATH} \
  ${PYTHON_CMD}
