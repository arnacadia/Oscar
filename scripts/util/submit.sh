#PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/

# VIRKAR EKKI #PYTHON=/home/staff/atli/.conda/envs/pygpu/bin/python
PYTHON=$OSSIAN/env/bin/python
#PYTHON=python

## Generic script for submitting any Theano job to GPU
# usage: submit.sh [scriptname.py script_arguments ... ]


## location of this script:
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"

cuda_dir="cuda"

gpu_id=$(python $THIS_DIR/gpu_lock.py --id-to-hog)
echo "#################################################"
echo $gpu_id
echo $THIS_DIR

# BUG: For some reason, I always get gpu_id=-1 now, don't know why
# for that reason I changed -1 to -2 below and it seems to work
if [ $gpu_id -gt -2 ]; then
    THEANO_FLAGS="device=cuda"
    #THEANO_FLAGS="cuda.root=/usr/local/cuda/bin,mode=FAST_RUN,device=cuda$gpu_id,floatX=float32,on_unused_input=ignore,dnn.include_path=/usr/local/$cuda_dir/include,dnn.library_path=/usr/local/$cuda_dir/lib64"
    #THEANO_FLAGS=""
    export THEANO_FLAGS
    $PYTHON $@

    python $THIS_DIR/gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'
    echo $gpu_id

fi
