#!/usr/bin/env bash
python -m sc2ai.run_agent --map MoveToBeacon --norender --step_mul 64 --parallel 3 --max_episodes 3000
