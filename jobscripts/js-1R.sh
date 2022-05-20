#!/bin/bash --login

# This is a script to submit simple.sh to SGE.
# Lines starting with # are comments
# Lines starting with #$ are instructions for SGE

# Tell SGE to run the job in the current directory
#$ -cwd

# Define a name for the job (default is name of jobscript)
#$ -N log1R

#$ -l mem512                   # For 16GB per core, any of the CPU types below
                               # Jobs will run on Haswell CPUs (AVX,AVX2 capable), max 16 cores

#$ -m bea
#$ -M helena.raymond-hayling@manchester.ac.uk
# Activate virtual env - these commands can be run inside a jobscript or an interactive session

module load apps/binapps/anaconda3/2021.11  # Python 3.9.7

source activate fta-env        # can also use 'conda activate test_env'
                                # but only on the login node. In
                                # jobscripts you should use 'source'
# Load the version you require


data="9am-1R"

minirun=0 #0=false
a=0.5
b=1
c=1.5
T=2
python3 a1_fibtrackMain.py $data $minirun $a $b $c $T
echo 'complete 1/3'

python3 a2_fibtrackStats.py $data $minirun $a $b $c $T
echo 'complete 2/3'

python3 b2_volumeRendering.py $data $minirun $a $b $c $T
echo 'complete 3/3'
