#!/usr/bin/env bash
python -m sc2ai.run_agent \
--map DefeatRoaches \
--norender \
--load_model \
--step_mul 8 \
--parallel 4 \
--gamma 0.95 \
--td_lambda 0.95 \
