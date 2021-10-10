#!/usr/bin/env bash

# Worker (signle-node and single GPU) for GPT-NeoX pretraining

CURR_DIR_PATH="$( cd -- "$( dirname "$( realpath "$0" ) " )" > /dev/null 2>&1 || exit ; pwd -P)"
source "${CURR_DIR_PATH}/paths.sh"

export MODULEPATH="/opt/lmod/stable/lmod/lmod/modulefiles:/lus/theta-fs0/software/environment/thetagpu/lmod/modulefiles:/lus/theta-fs0/software/spack/share/spack/modules/linux-ubuntu20.04-x86_64"
eval "$( /opt/lmod/stable/lmod/lmod/libexec/lmod load conda/2021-06-26 )"

conda activate ~/envs/gpt-neox
python -u /gpfs/mira-home/xduan7/projects/scalable-transformer/third-party/gpt-neox/pretrain_gpt2.py \
  --deepspeed_config "${NEOX_CONFIG_DIR_PATH}/deepspeed.json" \
  --megatron_config "${NEOX_CONFIG_DIR_PATH}/megatron.json"