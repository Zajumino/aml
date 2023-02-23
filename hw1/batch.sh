#!/bin/bash

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/hw1_%j_stdout.txt
#SBATCH --error=results/hw1_%j_stderr.txt
#SBATCH --time=00:20:00
#SBATCH --job-name=hw1_test
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw1
#SBATCH --array=0-159
#
#################################################

# set up env
. /home/fagg/tf_setup.sh
conda activate tf

# run exp
python hw1.py --exp_type 'bmi' --exp_index $SLURM_ARRAY_TASK_ID --epochs 100 --activation_hidden 'swish'
