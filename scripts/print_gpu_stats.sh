#!/usr/bin/env bash

# Gather and print GPU stats from all the nodes defined in `HOSTFILE_PATH`

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

# This nvidia-smi query command prints three lines for three GPU stats:
# - power utilization in percentage
# - GPU utilization in percentage
# - Memory utilization in percentage
# All the stats are averaged over all the GPUs (on the node that execute this query)
NVIDIA_SMI_QUERY_CMD="nvidia-smi \
  --query-gpu=power.draw,power.limit,utilization.gpu,utilization.memory \
  --format=csv | \
  awk -F',' \
    'NR>1 {pwr_draw+=\$1; pwr_limit+=\$2; gpu_util+=\$3; mem_util+=\$4; ++n} \
    END {print pwr_draw/pwr_limit; print gpu_util/n; print mem_util/n}' \
"

HOSTFILE_PATH="/Users/xduan7/.hostfile"
NUM_NODES=0
PWR_UTIL=0
GPU_UTIL=0
MEM_UTIL=0
while read -r NAME SLOT_INFO ; do
  if [[ -n "${NAME}" ]]; then
    echo "Gathering GPU stats from ${NAME} ..."
    if [[ "${NAME}" == "$( hostname )" ]]; then
      read -r -d '\n' PWR_UTIL_ GPU_UTIL_ MEM_UTIL_ <<< "$( eval "${NVIDIA_SMI_QUERY_CMD}" )"
    else
      read -r -d '\n' PWR_UTIL_ GPU_UTIL_ MEM_UTIL_ <<< "$( ssh ${NAME} "${NVIDIA_SMI_QUERY_CMD}" )"
    fi
    NUM_NODES=$((NUM_NODES + 1))
    PWR_UTIL=$((PWR_UTIL + PWR_UTIL_))
    GPU_UTIL=$((GPU_UTIL + GPU_UTIL_))
    MEM_UTIL=$((MEM_UTIL + MEM_UTIL_))
  fi
done < "${HOSTFILE_PATH}"
PWR_UTIL=$((PWR_UTIL / NUM_NODES))
GPU_UTIL=$((GPU_UTIL / NUM_NODES))
MEM_UTIL=$((MEM_UTIL / NUM_NODES))
printf "Averaged Power Utilization:\t %s%%\n" ${PWR_UTIL}
printf "Averaged GPU Utilization:\t %s%%\n" ${GPU_UTIL}
printf "Averaged Memory Utilization:\t %s%%\n" ${MEM_UTIL}
