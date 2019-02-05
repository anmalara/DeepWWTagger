#!/bin/bash

#SBATCH --partition=all
#SBATCH --time=5:00:00                  # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --constraint=GPU
#SBATCH --workdir   /beegfs/desy/user/amalara/output_varariables/workdir
#SBATCH --job-name  Sequential
#SBATCH --output    log_Sequential/sbatch-%N-%j.out  # File to which STDOUT will be written
#SBATCH --error     log_Sequential/sbatch-%N-%j.err  # File to which STDERR will be written
#SBATCH --mail-type ALL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user andrea.malara@desy.de  # Email to which notifications will be sent


source ~/.setpaths

cd /beegfs/desy/user/amalara/DeepWWTagger/code
python Sequential.py
