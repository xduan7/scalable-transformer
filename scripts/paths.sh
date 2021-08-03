#!/usr/bin/env bash

CURR_DIR_PATH="$(cd -- "$(dirname "$(realpath "$0")")" > /dev/null 2>&1 || exit ; pwd -P)"
PROJECT_DIR_PATH="$(realpath "${CURR_DIR_PATH}/../")"
CONDA_SH_PATH="${HOME}/software/anaconda3/etc/profile.d/conda.sh"
CONTAINER_PATH=/lus/theta-fs0/software/thetagpu/nvidia-containers/pytorch/pytorch_20.12-py3.simg


########################################################################
# Data directories
########################################################################

DATA_DIR_PATH="${PROJECT_DIR_PATH}/data"

RAW_DATA_DIR_PATH="${DATA_DIR_PATH}/raw"
INTERIM_DATA_DIR_PATH="${DATA_DIR_PATH}/interim"
PROCESSED_DATA_DIR_PATH="${DATA_DIR_PATH}/processed"

# English Wikipedia paths
ENWIKI_VOCAB_DIR_PATH="${RAW_DATA_DIR_PATH}/enwiki_vocab"
ENWIKI_TEXT_DIR_PATH="${PROCESSED_DATA_DIR_PATH}/enwiki_text"

BERT_CASED_ENWIKI_VOCAB_PATH="${ENWIKI_VOCAB_DIR_PATH}/bert-large-cased-vocab.txt"
BERT_UNCASED_ENWIKI_VOCAB_PATH="${ENWIKI_VOCAB_DIR_PATH}/bert-large-uncased-vocab.txt"
GPT2_ENWIKI_VOCAB_PATH="${ENWIKI_VOCAB_DIR_PATH}/gpt2-vocab.json"
GPT2_ENWIKI_MERGES_PATH="${ENWIKI_VOCAB_DIR_PATH}/gpt2-merges.txt"
T5_CASED_ENWIKI_VOCAB_PATH="${ENWIKI_VOCAB_DIR_PATH}/t5-cased-vocab.txt"
T5_UNCASED_ENWIKI_VOCAB_PATH="${ENWIKI_VOCAB_DIR_PATH}/t5-uncased-vocab.txt"

BERT_CASED_ENWIKI_TEXT_DIR_PATH="${ENWIKI_TEXT_DIR_PATH}/bert_cased"
BERT_UNCASED_ENWIKI_TEXT_DIR_PATH="${ENWIKI_TEXT_DIR_PATH}/bert_uncased"
GPT2_ENWIKI_TEXT_DIR_PATH="${ENWIKI_TEXT_DIR_PATH}/gpt2"
T5_CASED_ENWIKI_TEXT_DIR_PATH="${ENWIKI_TEXT_DIR_PATH}/t5_cased"
T5_UNCASED_ENWIKI_TEXT_DIR_PATH="${ENWIKI_TEXT_DIR_PATH}/t5_uncased"


########################################################################
# Third-party directories
########################################################################

THIRD_PARTY_DIR_PATH="${PROJECT_DIR_PATH}/third-party"

MEGATRON_DIR_PATH="${THIRD_PARTY_DIR_PATH}/Megatron-LM"
MEGATRON_CHECKPOINT_DIR_PATH="${MEGATRON_DIR_PATH}/checkpoints"
