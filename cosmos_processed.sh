#!/bin/bash
# This is an example job script for running a serial program
# these lines are comments
# SLURM directives are shown below

# Configure the resources needed to run my job, e.g.

# job name (default: name of script file)
#SBATCH --job-name=process_cosmos
# resource limits: cores, max. wall clock time during which job can be running
# and maximum memory (RAM) the job needs during run time:

#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --mem=20G


# define log files for output on stdout and stderr
#SBATCH --output=cosmos_processed.out

# choose system/queue for job submission (default: sciama2.q)
# for more information on queues, see related articles
#SBATCH --partition=sciama3.q

# set up the software environment to run your job in
# first remove all pre-loaded modules to avoid any conflicts
module purge
# load your system module e.g.
module load system/intel64
# now load all modules (e.g. libraries or applications) that are needed
# to run the job, e.g.
module load intel_comp/2019.2
module load cpython/3.7.1

# now execute all commands/programs that you want to run sequentially, e.g.
cd /users/psudek/measuring_luminosity_function/


time /opt/apps/pkgs/anaconda3/2019.03/intel64/bin/python COSMOS_Processed_catalogue.py
