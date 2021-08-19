#!/usr/bin/env bash

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
PROJECT_DIR_PATH="$( realpath "${CURR_DIR_PATH}/../" )"


########################################################################
# Executable paths
########################################################################

SRC_DIR_PATH="${PROJECT_DIR_PATH}/src"
THIRD_PARTY_DIR_PATH="${PROJECT_DIR_PATH}/third-party"

MEGATRON_DIR_PATH="${THIRD_PARTY_DIR_PATH}/megatron"
MEGATRON_DS_DIR_PATH="${THIRD_PARTY_DIR_PATH}/deepseed-examples/Megatron-LM-v1.1.5-ZeRO3"


########################################################################
# Data paths
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
# Checkpoint paths
########################################################################

CHECKPOINT_DIR_PATH="${PROJECT_DIR_PATH}/checkpoints"

# Original Megatron-LM checkpoints
MEGATRON_CHECKPOINT_DIR_PATH="${CHECKPOINT_DIR_PATH}/megatron"
MEGATRON_BERT_CASED_345M_CHECKPOINT_DIR_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/bert_cased_345m"
MEGATRON_BERT_UNCASED_345M_CHECKPOINT_DIR_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/bert_uncased_345m"
MEGATRON_GPT_345M_CHECKPOINT_DIR_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/gpt_345m"
MEGATRON_BERT_CASED_345M_CHECKPOINT_DIR_PATH="${MEGATRON_CHECKPOINT_DIR_PATH}/bert_cased_345m"

# Retrained checkpoints
RETRAINED_CHECKPOINT_DIR_PATH="${CHECKPOINT_DIR_PATH}/scalable-transformer"
RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_cased_345m"
RETRAINED_BERT_UNCASED_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_uncased_345m"
RETRAINED_GPT_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/gpt_345m"
RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_cased_345m"


########################################################################
# Configuration paths
########################################################################

CONFIG_DIR_PATH="${CURR_DIR_PATH}/config"

DS_ZERO2_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero2.json"
DS_ZERO3_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero3.json"


########################################################################
# Other paths
########################################################################

LOG_DIR_PATH="${PROJECT_DIR_PATH}/logs"

CONDA_PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/"

CONTAINER_PATH="/lus/theta-fs0/projects/candle_aesp/xduan7/containers/megatron_ds.simg"
CONTAINER_PYTHON_PATH="/opt/conda/bin/python3"

