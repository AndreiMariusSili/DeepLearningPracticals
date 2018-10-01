#!/bin/sh
#PBS -lwalltime=06:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module load NCCL

export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

mkdir "$TMPDIR"/"$(whoami)"

python part1/sequence_length_experiment.py --model_path ""$TMPDIR"/$(whoami)/part3/"

cp -r "$TMPDIR"/"$(whoami)"/part1/ /home/$(whoami)/part1/models/