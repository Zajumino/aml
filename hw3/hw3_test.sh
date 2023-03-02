#!/bin/bash
#
#SBATCH --partition=debug
# memory in MB
#SBATCH --mem=15000
#SBATCH --output=results/hw3_%04a_stdout.txt
#SBATCH --error=results/hw3_%04a_stderr.txt
#SBATCH --time=00:10:00
#SBATCH --job-name=hw3_test
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw3
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# 2023
python hw3.py @oscer.txt @exp.txt @net_shallow.txt --epochs 10 --exp_index 1

