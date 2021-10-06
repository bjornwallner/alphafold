#!/bin/bash
#SBATCH -n 1 -c 8
###SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH -A snic2021-5-373
#SBATCH -t 960



module load Python/3.7.0-anaconda-5.3.0-extras-nsc1
module load buildenv-gcccuda/.11.1-9.3.0-bare
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/sse/manual/CUDA/11.2.1_460.32.03/lib64/:/proj/wallner/cuda11.2_cudnn8//lib64/

source activate /proj/wallner/users/x_bjowa/.conda/envs/alphafold/
which python
echo Running CMD: python /proj/wallner/apps/alphafold/run_alphafold.py $@
python /proj/wallner/apps/alphafold/run_alphafold.py $@
