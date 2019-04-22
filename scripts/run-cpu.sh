#!/usr/bin/env bash
#
#  Execute from the current working directory
#$ -cwd
#
#  Concurrent threads
####$ -pe smp 4
#
#  This is a long-running job
#$ -l inf
#
#  Can use up to 64GB of memory
#$ -l vf=4G
#
usemini3(){
  #export PATH=/home/jl84/miniconda3/bin:$PATH
  # added by Miniconda3 4.5.12 installer
  # >>> conda init >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$(CONDA_REPORT_ERRORS=false '/home/jl84/miniconda3/bin/conda' shell.bash hook 2> /dev/null)"
  if [ $? -eq 0 ]; then
      \eval "$__conda_setup"
  else
      if [ -f "/home/jl84/miniconda3/etc/profile.d/conda.sh" ]; then
          . "/home/jl84/miniconda3/etc/profile.d/conda.sh"
          CONDA_CHANGEPS1=false conda activate base
      else
          \export PATH="/home/jl84/miniconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda init <<<
}

usemini3
export TEMP=/research/xai/jl84/tmp
source activate sc2cpu

IND=$SGE_TASK_ID
pwd

python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
echo $1
python ../src/run_worker.py --redis-address $1:10001 --ray-num-cpus 1 --temp-dir $TEMP/$IND
