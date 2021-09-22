#!/usr/bin/env bash

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

# Generate hostfile
source "${CURR_DIR_PATH}/generate_hostfile.sh"


PYTHON_CMD="${NEOX_DIR_PATH}/deepy.py \
  ${NEOX_DIR_PATH}/pretrain_gpt2.py \
  -d ${NEOX_CONFIG_DIR_PATH} local_setup.yml 345m.yml \
"

# shellcheck disable=SC2086
# --env NCCL_DEBUG=WARN,LD_LIBRARY_PATH="/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" \
# --bind /gpfs,/lus,/usr/local/cuda/lib64 \
singularity exec \
  --nv \
  --cleanenv \
  --env NCCL_DEBUG=WARN \
  --bind /gpfs,/lus \
  ${NEOX_CONTAINER_PATH} \
  ${PYTHON_CMD}
