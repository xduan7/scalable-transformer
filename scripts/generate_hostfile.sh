#!/usr/bin/env bash

# Construct a new hostfile compatible with DeepSpeed, stored in
# environment variable `HOSTFILE_PATH, defined in `paths.sh`

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

echo "Generating hostfile ${HOSTFILE_PATH} from COBALT_NODEFILE ..."

if [[ -f "${COBALT_NODEFILE}" ]]; then
  rm -f "${HOSTFILE_PATH}" 2> /dev/null && touch "${HOSTFILE_PATH}"
  while read -r NAME; do
    # IP_ADDR_CMD="hostname -I | cut -d' ' -f1"
    NUM_SLOTS_CMD="nvidia-smi --list-gpus | wc -l"
    if [[ "${NAME}" == "$( hostname )" ]]; then
      # ADDR=$( eval "${IP_ADDR_CMD}" )
      NUM_SLOTS=$( eval "${NUM_SLOTS_CMD}" )
    else
      # ADDR=$( ssh -n "${NAME}" "${IP_ADDR_CMD}" )
      NUM_SLOTS=$( ssh -n "${NAME}" "${NUM_SLOTS_CMD}" )
    fi
    echo "${NAME} slots=${NUM_SLOTS}" >> "${HOSTFILE_PATH}"
  done < "${COBALT_NODEFILE}"
fi
