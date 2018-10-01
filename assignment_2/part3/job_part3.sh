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

python part3/train.py --txt_file "/home/$(whoami)/part3/data/book_EN_democracy_in_the_US.txt"\
                      --model_path "/home/$(whoami)/part3/models/"\
                      --train_steps "100000"\
                      --dropout_keep_prob "0.5"\
                      --learning_rate_decay "0.90"