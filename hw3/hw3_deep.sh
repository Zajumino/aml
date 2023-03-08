#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/hw3_%04a_stdout.txt
#SBATCH --error=results/hw3_%04a_stderr.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=hw3_deep
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw3
#SBATCH --array=0-4
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
python hw3.py @oscer.txt @exp.txt @net_deep.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --save_model

