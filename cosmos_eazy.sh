#!/bin/bash
# This is an example job script for running a serial program
# these lines are comments
# SLURM directives are shown below

# Configure the resources needed to run my job, e.g.

# job name (default: name of script file)
#SBATCH --job-name=eazy_cosmos
# resource limits: cores, max. wall clock time during which job can be running
# and maximum memory (RAM) the job needs during run time:

#SBATCH --ntasks=32
#SBATCH --time=50:00:00
#SBATCH --mem=40G



# define log files for output on stdout and stderr
#SBATCH --output=cosmos_eazy.out

# choose system/queue for job submission (default: sciama2.q)
# for more information on queues, see related articles
#SBATCH --partition=sciama4.q

# set up the software environment to run your job in
# first remove all pre-loaded modules to avoid any conflicts
module purge
# load your system module e.g.
module load system/intel64
# now load all modules (e.g. libraries or applications) that are needed
# to run the job, e.g.
module load intel_comp/2019.2

source /users/psudek/virtual_environments/eazy_env/bin/activate
# now execute all commands/programs that you want to run sequentially, e.g.
cd /users/psudek/measuring_luminosity_function


time ~/virtual_environments/eazy_env/bin/python cosmos_eazy.py $SLURM_NTASKS
