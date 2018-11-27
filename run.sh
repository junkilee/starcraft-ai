#!/usr/bin/env bash
python -m sc2ai.run_agent \
--map MoveToBeacon \
--norender \
--step_mul 16 \
--parallel 5 \
--gamma 0.9 \
--td_lambda 0.9 \
