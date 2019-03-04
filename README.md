# starcraft-ai

This repository contain the source code to develop explainable AI systems for playing the game Starcraft.

# Installation
`pip install trfl`

`pip install tensorflow-probability==0.5.0`

Maybe:
`pip install tensorflow-probability-gpu==0.5.0`



# Training an agent
To run agent, first navigate to project root directory then run:

`./bin/run`

## Scheduling jobs on the grid
To schedule a job on the brown grid, navigate to the project root directory and run:

`qsub -now y ./bin/grid-run`

This will run ./bin/run/ on the grid
