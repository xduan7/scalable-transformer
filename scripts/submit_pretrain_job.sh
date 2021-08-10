#!/usr/bin/env bash

# Arguments:
#   pretrain script name: positional argument; must exist in `./`
#   -d or --distributed: enable distributed training (for multi-node)
#
# Usage:
#   Single-GPU training: `./submit_pretrain_job.sh pretrain_megatron_bert_cased_345m.sh`
#   Multi-node training: `./submit_pretrain_job.sh pretrain_megatron_bert_cased_345m.sh -d`

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

PRETRAIN_SCRIPT_PATH="${CURR_DIR_PATH}/${1}"
COBALT_JOB_NAME="${1}${2}"

if  [ "$2" = "-d" ] || [ "$2" = "--distributed" ]; then
  PRETRAIN_CMD="$( which mpirun ) \
    -hostfile ${COBALT_NODEFILE} \
    --map-by ppr:1:node \
    ${PRETRAIN_SCRIPT_PATH} ${2} \
  "
else
  PRETRAIN_CMD="${PRETRAIN_SCRIPT_PATH} ${2}"
fi

qsub \
  -n 8 \
  -t 6:00:00 \
  -A CVD-Mol-AI \
  -q full-node \
  -O "${LOG_DIR_PATH}/${COBALT_JOB_NAME}" \
  "${PRETRAIN_CMD}"
