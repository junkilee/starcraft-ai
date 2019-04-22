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
#$ -l vf=32G
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
source activate sc2gpu

nvidia-smi

#IND=$SGE_TASK_ID
pwd
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=32
export IP=`python -c "import socket; print(socket.gethostbyname(socket.gethostname()))"`
echo $IP
ray start --head --redis-port 10001 --num-cpus 3 --num-gpus 1 --temp-dir $TEMP
python ../src/run_gpu.py --redis-address $IP:10001 --queue-trials --num-workers=$NUM_WORKERS
