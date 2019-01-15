from glob import glob
import os
import sys
from variables import *
sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from parallelise import *

@timeit
def ClearDirectory(dir):
    list_processes  = []
    for dir_ in glob(dir+"/*/*"):
        list_processes.append( ["python", "ClearDirectory.py", str(dir_+"/"), str(1)] )
    print len(list_processes)
    parallelise(list_processes, 20)

@timeit
def DeleteFiles(dir):
    for file in glob(dir+"/*"):
        os.remove(file)


try:
    dir  = sys.argv[1]
except:
    dir  = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/input_varariables/NTuples_Tagger"
try:
    flag = int(sys.argv[2])
except :
    flag = 0

print flag, dir

if not flag:
    print "ClearDirectory"
    ClearDirectory(dir)
else:
    print "DeleteFiles"
    DeleteFiles(dir)
