#!/bin/bash --login

# This is a script to submit simple.sh to SGE.
# Lines starting with # are comments
# Lines starting with #$ are instructions for SGE

#$ -cwd                         # Tell SGE to run the job in the current directory
#$ -N abc-routine-2             # Define a name for the job (default is name of jobscript)

#$ -m bea
#$ -M helena.raymond-hayling@manchester.ac.uk

#$-pe smp.pe 6                 #Using 6 cores

#$ -l mem256                   # For 16GB per core, any of the CPU types below
                               # Jobs will run on Haswell CPUs (AVX,AVX2 capable), max 16 cores



source activate fta-env        # Activate virtual env - these commands can be run inside a jobscript or an interactive session
                                # jobscripts you should use 'source'

module load apps/binapps/anaconda3/2020.07  # Python 3.8.3

python b3_ABC_routine.py
