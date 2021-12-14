#!/bin/bash --login

# This is a script to submit simple.sh to SGE.
# Lines starting with # are comments
# Lines starting with #$ are instructions for SGE

# Tell SGE to run the job in the current directory
#$ -cwd

# Define a name for the job (default is name of jobscript)
#$ -N stats-fta

#$ -l mem256                   # For 16GB per core, any of the CPU types below
                               # Jobs will run on Haswell CPUs (AVX,AVX2 capable), max 16 cores

#$ -m bea
#$ -M helena.raymond-hayling@manchester.ac.uk
# Activate virtual env - these commands can be run inside a jobscript or an interactive session

source activate fta-env        # can also use 'conda activate test_env'
                                # but only on the login node. In
                                # jobscripts you should use 'source'
# Load the version you require
module load apps/binapps/anaconda3/2020.07  # Python 3.8.3


python a2_fibtrackStats.py
# python 3_mechanicalLoading.py
# python 4_volumeRendering.py
