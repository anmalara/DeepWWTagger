#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_STORED

python selectBranches.py $1 $2 $3 $4
