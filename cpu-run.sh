#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a long-running job
#$ -l inf
#
#  Can use up to 64GB of memory
#$ -l vf=16G
#
#$ -pe smp 8

export SC2PATH=/research/xai/starcraft/users/ble2/StarCraftII
source /research/xai/starcraft/users/ble2/miniconda3/bin/activate sc2

python -m sc2ai.run_agent \
--map DefeatZerglingsAndBanelings \
--noload_model \
--norender \
--noepsilon \
--nocuda \
--step_mul 8 \
--parallel 8 \
--gamma 0.95 \
--td_lambda 0.95 \
