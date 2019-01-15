#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_STORED

python preProcessing_Conv2D_variables.py $1 $2 $3 $4 $5 $6
