#!/bin/bash

#SBATCH --partition=maxwell
#SBATCH --time=10:00:00                 # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /beegfs/desy/user/amalara/code/
#SBATCH --job-name  JetImage
#SBATCH --output    log_JetImage/sbatch-%N-%j.out  # File to which STDOUT will be written
#SBATCH --error     log_JetImage/sbatch-%N-%j.err  # File to which STDERR will be written
#SBATCH --mail-type ALL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user andrea.malara@desy.de  # Email to which notifications will be sent


python JetImage.py
