#!/usr/bin/env bash

# python runner.py --map MoveToBeacon --agent sc2ai.learned_agents.RoachesAgent --norender
#python -m sc2ai.run_agent --map MoveToBeacon --agent sc2ai.learned_agents.RoachesAgent --norender --step_mul 64 --parallel 3
python -m sc2ai.run_agent --map MoveToBeacon --agent sc2ai.parallel_agent.ParallelAgent --norender --step_mul 64 --parallel 5 --max_episodes 10000
