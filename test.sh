#!/bin/bash
# Specify a job name
#$ -N test
# Project name and target queue
#$ -P papiez.prjc
#$ -q short.qc
# Run the job in the current working directory
#$ -cwd -j y
# Log locations which are relative to the current
# working directory of the submission
#$ -o /well/papiez/users/zou393/Documents/DPhil/script/log-test
###$ -e error.log
# Parallel environemnt settings
#  For more information on these please see the wiki
#  Allowed settings:
#   shmem
#   mpi
#   node_mpi
#   ramdisk
#$ -pe shmem 1

# Some useful data about the job to help with debugging
echo "------------------------------------------------"
echo "SGE Job ID: $JOB_ID"
echo "SGE Task ID: $SGE_TASK_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
# Begin writing your script here
exefni_ac=/well/papiez/users/zou393/Documents/DPhil/script/train.py
hupfni_ac=/well/papiez/users/zou393/Documents/DPhil/script/log-test.out
config=/well/papiez/users/zou393/Documents/DPhil/script/test-job.json

module load Anaconda3/5.1.0
source activate /users/papiez/zou393/miniconda3/envs/mimic
module load cuda/10.0
export MPLBACKEND=TkAgg


python -u $exefni_ac -c $config
# End of job script
echo "------------------------------------------------"
echo "SGE Job ID: $JOB_ID"
echo "SGE Task ID: $SGE_TASK_ID"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "End at: "`date`
echo "------------------------------------------------"
