#!/usr/bin/env bash

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
PROJECT_DIR_PATH="$( realpath "${CURR_DIR_PATH}/../" )"


########################################################################
# Executable paths
########################################################################

SRC_DIR_PATH="${PROJECT_DIR_PATH}/src"
SCRIPT_DIR_PATH="${PROJECT_DIR_PATH}/scripts"
THIRD_PARTY_DIR_PATH="${PROJECT_DIR_PATH}/third-party"

MEGATRON_DIR_PATH="${THIRD_PARTY_DIR_PATH}/megatron"
MEGATRON_DS_DIR_PATH="${THIRD_PARTY_DIR_PATH}/deepseed-examples/Megatron-LM-v1.1.5-ZeRO3"
# MEGATRON_DS_DIR_PATH="${THIRD_PARTY_DIR_PATH}/deepseed-examples/Megatron-LM-v1.1.5-3D_parallelism"
# MEGATRON_DS_DIR_PATH="${THIRD_PARTY_DIR_PATH}/magatron-deepspeed"

NEOX_DIR_PATH="${THIRD_PARTY_DIR_PATH}/gpt-neox"

PDSH_PATH="$( realpath "${HOME}/bin" )"
PDSH_LIB_PATH="$( realpath "${HOME}/lib/pdsh" )"
DEEPSPEED_PATH="$( realpath "${HOME}/.local/bin/deepspeed" )"


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

# Retrained checkpoints
RETRAINED_CHECKPOINT_DIR_PATH="${CHECKPOINT_DIR_PATH}/scalable-transformer"
RETRAINED_BERT_CASED_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_cased_345m"
RETRAINED_BERT_UNCASED_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_uncased_345m"
RETRAINED_GPT_345M_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/gpt_345m"

RETRAINED_BERT_CASED_175B_CHECKPOINT_DIR_PATH="${RETRAINED_CHECKPOINT_DIR_PATH}/bert_cased_175b"


########################################################################
# Configuration paths
########################################################################

CONFIG_DIR_PATH="${CURR_DIR_PATH}/config"

DS_ZERO0_MINIMAL_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero0_minimal.json"
DS_ZERO3_MINIMAL_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero3_minimal.json"
DS_ZERO3_OFFLOAD_MINIMAL_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero3_offload_minimal.json"
DS_ZERO3_OFFLOAD_RELEASE_CONFIG_PATH="${CONFIG_DIR_PATH}/ds_zero3_offload_release.json"

NEOX_CONFIG_DIR_PATH="${CONFIG_DIR_PATH}/neox"
NEOX_345M_DIR_PATH="${NEOX_CONFIG_DIR_PATH}/345m.yml"
NEOX_LOCAL_SETUP_CONFIG_PATH="${NEOX_CONFIG_DIR_PATH}/local_setup.yml"


########################################################################
# Other paths
########################################################################

LOG_DIR_PATH="${PROJECT_DIR_PATH}/logs"

CONDA_PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/"

MEGATRON_DS_CONTAINER_PATH="/lus/theta-fs0/projects/candle_aesp/xduan7/containers/megatron_ds.simg"
MEGATRON_DS_CONTAINER_PYTHON_PATH="/opt/conda/bin/python3"
NEOX_CONTAINER_PATH="/lus/theta-fs0/projects/candle_aesp/xduan7/containers/gpt-neox.simg"
NEOX_CONTAINER_PYTHON_PATH="/usr/bin/python"

HOSTFILE_PATH="$( realpath "${SCRIPT_DIR_PATH}/.hostfile" )"
