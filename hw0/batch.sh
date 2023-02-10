#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#
#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/hw0_%j_stdout.txt
#SBATCH --error=results/hw0_%j_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=hw0_test
#SBATCH --mail-user=ikang@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504306/aml/hw0
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python hw0.py --epochs 10 --exp 0

