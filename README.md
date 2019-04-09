# starcraft-ai

This repository contain the source code to develop explainable AI systems for playing the game Starcraft.

# Installation

** Note that currently Brown CS GPUs are not supported by Tensorflow. Use CPU version instead **

## Create new Conda Environment (CPU or GPU)
`conda create -n sc-cpu python=3.6`  
`conda create -n sc-gpu python=3.6`

## ###################################

## CPU Setup Environment
`source activate sc-cpu`  
`pip install -r cpu-requirements.txt`  
`./bin/setup`  

## GPU Setup Environment
`source activate sc-gpu`  
`pip install -r gpu-requirements.txt`  
`./bin/setup`  

# Training an agent
To run agent, first navigate to project root directory then run:

`./bin/run`

## Scheduling jobs on the grid
To schedule a job on the brown grid, navigate to the project root directory and run:

`qsub -now y ./bin/grid-run`

This will run ./bin/run/ on the grid.
