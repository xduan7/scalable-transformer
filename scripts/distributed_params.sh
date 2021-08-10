#!/usr/bin/env bash

MICRO_BATCH=4
TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2
MASTER_PORT=6000

# Get the MPI specifications using Python
# Note that there might be warning/error messages during import
module load conda
conda activate base
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

GPUS_PER_NODE=$( nvidia-smi --list-gpus | wc -l )
WORLD_SIZE=$(("${GPUS_PER_NODE}*${NNODES}"))
GLOBAL_BATCH=$(("${MICRO_BATCH}*${GPUS_PER_NODE}*${NNODES}"))

DISTRIBUTED_MODULE="-m torch.distributed.launch"
DISTRIBUTED_MODULE_ARGS="\
  --nproc_per_node ${GPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port 6000 \
"
DISTRIBUTED_ARGS="\
  --tensor-model-parallel-size ${TENSOR_MP_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_MP_SIZE} \
  --DDP-impl torch \
"
BERT_BATCH_ARGS="\
  --micro-batch-size ${MICRO_BATCH} \
  --global-batch-size ${GLOBAL_BATCH} \
"
