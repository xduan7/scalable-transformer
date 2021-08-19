#!/usr/bin/env bash

# Arguments:
#   pretrain script name: positional argument; must exist in `./`
#   NUM_NODES: number of nodes for pre-training
#
# Usage:
#   # training on N nodes
#   $ ./submit_pretrain_job.sh \
#       pretrain_megatron_bert_cased_345m.sh N


CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

PRETRAIN_SCRIPT_PATH="${CURR_DIR_PATH}/${1}"
PRETRAIN_LOG_DIR_PATH="${LOG_DIR_PATH}/${1%.*}_on_${2}_nodes"

if [[ "${2}" == 1 ]]; then
  PRETRAIN_CMD="${PRETRAIN_SCRIPT_PATH} -d"
else
  PRETRAIN_CMD="\
    module load openmpi &&
    mpirun \
      --hostfile ${COBALT_NODEFILE} \
      --map-by ppr:1:node \
      ${SINGULARITY_CMD} ${PRETRAIN_SCRIPT_PATH} -d \
  "
fi

# shellcheck disable=SC2086
qsub \
  -n ${2} \
  -t 6:00:00 \
  -A CVD-Mol-AI \
  -q full-node \
  -O ${PRETRAIN_LOG_DIR_PATH} \
  ${PRETRAIN_CMD}