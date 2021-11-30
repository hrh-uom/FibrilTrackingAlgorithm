#!/bin/bash --login

# This is a script to submit simple.sh to SGE.
# Lines starting with # are comments
# Lines starting with #$ are instructions for SGE

# Tell SGE to run the job in the current directory
#$ -cwd

# Define a name for the job (default is name of jobscript)
#$ -N fta

#$ -m bea
#$ -M helena.raymond-hayling@manchester.ac.uk

#$ -l mem256                   # For 16GB per core, any of the CPU types below
                               # Jobs will run on Haswell CPUs (AVX,AVX2 capable), max 16 cores

                               # Activate virtual env - these commands can be run inside a jobscript or an interactive session

source activate fta-env # jobscripts you should use 'source'

# Load the version you require
module load apps/binapps/anaconda3/2020.07  # Python 3.8.3

python 1_fibtrackMain.py
