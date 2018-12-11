#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#  This is a gpu job
#$ -l gpus=1
#
#  This is a long-running job
#$ -l inf
#
#  Can use up to 64GB of memory
#$ -l vf=64G
#

python -m sc2ai.run_agent \
--map DefeatRoaches \
--noload_model \
--norender \
--noepsilon \
--cuda \
--step_mul 8 \
--parallel 10 \
--gamma 0.95 \
--td_lambda 0.95 \
