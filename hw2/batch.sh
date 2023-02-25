#!/bin/bash

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/hw2_%j_stdout.txt
#SBATCH --error=results/hw2_%j_stderr.txt
#SBATCH --time=00:10:00
#SBATCH --job-name=hw2_test
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw2
#SBATCH --array=200-699
#
#################################################

# set up env
. /home/fagg/tf_setup.sh
conda activate tf

# run exp
python hw2.py --exp_type 'dropout' --cpus_per_task $SLURM_CPUS_PER_TASK --exp_index $SLURM_ARRAY_TASK_ID --epochs 1000 --activation_hidden 'swish'
