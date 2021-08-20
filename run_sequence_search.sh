#!/bin/bash
#SSSBATCH -n 1 -c 4


module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
#module load buildenv-gcccuda/.11.1-9.3.0-bare
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/sse/manual/CUDA/11.2.1_460.32.03/lib64/:/proj/wallner/cuda11.2_cudnn8//lib64/

source activate /proj/wallner/users/x_bjowa/.conda/envs/alphafold/
which python
echo Running CMD: python /proj/wallner/apps/alphafold/run_sequence_search.py $@
python /proj/wallner/apps/alphafold/run_sequence_search.py $@



