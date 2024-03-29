#!/usr/bin/env bash

# Get the following variables for distributed training:
# - (MPI and machine specifications)
#   - MASTER_ADDR
#   - NNODES
#   - NODE_RANK
#   - GPUS_PER_NODE
# - (pretraining command arguments or options)
#   - DISTRIBUTED_MODULE
#   - DISTRIBUTED_MODULE_ARGS
#   - DISTRIBUTED_ARGS
#   - BERT_BATCH_ARGS
#
# Note:
#   This script works for both single-node and multi-node mode.
#   For multi-node mode, this script should be executed within
#   `mpirun` or `mpiexec`; otherwise `mpi4py` module won't be able
#   to get the correct MPI specifications.

MICRO_BATCH_SIZE=4
NUM_MICRO_BATCH=1
TENSOR_MP_SIZE=1
PIPELINE_MP_SIZE=1

MASTER_PORT=6000

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

# Get the MPI specifications using Python
# Command like `module load` or 'conda' won't work in mpirun
# `NNODES` will always be 1 if not executed in mpirun
source "${CONDA_PATH}/bin/activate" base
# Note that there might be warning/error messages during the execution of the following code
GET_MPI_SPECS_FROM_MPI4PY=$(python3 -W ignore << EOF
import socket
from mpi4py import MPI
master_addr = MPI.COMM_WORLD.bcast(
    socket.gethostname() if MPI.COMM_WORLD.Get_rank() == 0 else None,
    root=0,
)
print(master_addr)
print(MPI.COMM_WORLD.Get_size())
print(MPI.COMM_WORLD.Get_rank())
EOF
)
MPI_SPECS=$( tail --lines 3 <<< "${GET_MPI_SPECS_FROM_MPI4PY}" )
read -r -d '\n' MASTER_ADDR NNODES NODE_RANK <<< "${MPI_SPECS}"

# Generate distributed parameters for pretraining
GPUS_PER_NODE=$( nvidia-smi --list-gpus | wc -l )
WORLD_SIZE=$((GPUS_PER_NODE*NNODES))
GLOBAL_BATCH_SIZE=$((NUM_MICRO_BATCH*MICRO_BATCH_SIZE*WORLD_SIZE/PIPELINE_MP_SIZE/TENSOR_MP_SIZE))

DISTRIBUTED_MODULE="-m torch.distributed.launch"
DISTRIBUTED_MODULE_ARGS="\
  --nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
"
DISTRIBUTED_ARGS="\
  --tensor-model-parallel-size ${TENSOR_MP_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_MP_SIZE} \
"
