#!/bin/bash

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/hw0_%j_stdout.txt
#SBATCH --error=results/hw0_%j_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=hw1_test
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw1
#
#################################################

# set up env
. /home/fagg/tf_setup.sh
conda activate tf

# run exp
python hw1.py --exp_type 'bmi'
