#!/usr/bin/env bash

# Arguments:
#   pretrain script name: positional argument; must exist in `./`
#   -d or --distributed: enable distributed training (for multi-node)
#
# Usage:
#   $ ./submit_pretrain_job.sh pretrain_megatron_bert_cased_345m.sh  # single GPU
#   or
#   $ ./submit_pretrain_job.sh pretrain_megatron_bert_cased_345m.sh -d # multiple nodes

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

PRETRAIN_SCRIPT_PATH="${CURR_DIR_PATH}/${1}"
COBALT_JOB_NAME="${1}${2}"

PRETRAIN_CMD="${PRETRAIN_SCRIPT_PATH} ${2}"
if  [ "${2}" = "-d" ] || [ "${2}" = "--distributed" ]; then
  NUM_NODES=8
  MPIRUN="$( which mpirun )"
  PRETRAIN_CMD="${MPIRUN} \
    --hostfile ${COBALT_NODEFILE} \
    --map-by ppr:1:node \
    ${PRETRAIN_CMD} \
  "
else
  NUM_NODES=1
fi

# shellcheck disable=SC2086
qsub \
  -n ${NUM_NODES} \
  -t 6:00:00 \
  -A CVD-Mol-AI \
  -q full-node \
  -O "${LOG_DIR_PATH}/${COBALT_JOB_NAME}" \
  ${PRETRAIN_CMD}
