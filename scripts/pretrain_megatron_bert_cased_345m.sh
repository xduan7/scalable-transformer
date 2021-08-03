#!/usr/bin/env bash

CURR_DIR_PATH="$(cd -- "$(dirname "$(realpath "$0")")" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/setup.sh"
source "${CURR_DIR_PATH}/paths.sh"

LOAD_CHECKPOINT_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/bert_cased_345m"
SAVE_CHECKPOINT_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/bert_cased_345m_retrained"
VOCAB_FILE="${BERT_CASED_ENWIKI_VOCAB_PATH}"
DATA_PATH="${BERT_CASED_ENWIKI_TEXT_DIR_PATH}/text_sentence"

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
  --lr-warmup-fraction 0.01 \
  --micro-batch-size 4 \
  --global-batch-size 8 \
  --vocab-file ${VOCAB_FILE} \
  --split 949,50,1 \
  --fp16 \
"

OUTPUT_ARGS="\
  --log-interval 10 \
  --save-interval 500 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations
"

PYTHON_CMD="python3 ${MEGATRON_DIR_PATH}/pretrain_bert.py \
  ${BERT_ARGS} \
  ${OUTPUT_ARGS} \
  --save ${SAVE_CHECKPOINT_PATH} \
  --load ${LOAD_CHECKPOINT_PATH} \
  --data-path ${DATA_PATH}"

# `PYTHON_CMD` must be split into globs for argparse parsing
# shellcheck disable=SC2086
singularity exec \
  --nv \
  --bind /gpfs/mira-home/xduan7,/lus/theta-fs0/projects/candle_aesp/xduan7/data \
  "${CONTAINER_PATH}" \
  ${PYTHON_CMD}
