# Running
* Make sure that the pygpu environment is activated (only that environment!)
* Don't know why, but I set PYTHON=$OSSIAN/env/bin/python in submit.sh and it seems to be the one that works ?
* Set `if [ $gpu_id -gt -2 ]; then` instead of `if [ $gpu_id -gt -1 ]; then` in submit.sh
* Set the following slurm parameters for gpu:
#SBATCH --gres=gpu:1
#SBATCH --nodelist=torpaq
