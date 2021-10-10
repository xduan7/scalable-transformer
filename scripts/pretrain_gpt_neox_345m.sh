#!/usr/bin/env bash

# Pre-train GPT-NeoX on multiple nodes
#
# Arguments:
#   -n: number of nodes for pretraining; using all the nodes in
#       the hostfile if not given

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"
#source "${CURR_DIR_PATH}/generate_hostfile.sh"

while getopts ":n:" opt; do
    case "${opt}" in
        n) NUM_NODES=${OPTARG};;
        *)  echo "Illegal argument."; exit 1;;
    esac
done
shift $((OPTIND-1))


if [ -z "$NUM_NODES" ]
then
    NUM_NODES="$( cat "${HOSTFILE_PATH}" | wc -l )"
fi

NUM_WORKERS=$((NUM_NODES*8))
mpirun -n ${NUM_WORKERS} -x MODULEPATH --map-by ppr:8:node -hostfile ${HOSTFILE_PATH} "${CURR_DIR_PATH}/pretrain_gpt_neox_345m_worker.sh"
