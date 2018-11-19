from variables import *

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from parallelise import *

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

####################
# TODO: check the subCounter in selectBranches
###################

try:
  min = int(sys.argv[1])
  max = int(sys.argv[2])
  step = int(sys.argv[3])
except:
  min = 0
  max = 10000
  step = 10


log_folder= "./log_selectBranches/"
filelist=glob.glob(log_folder+"*txt")
for file in filelist:
  os.remove(file)

if not os.path.exists(log_folder):
  os.makedirs(log_folder)

list_processes  = []
list_logfiles   = []
i = 0
for bkg in bkgs:
  for radius in radii:
    for file_min in range(min, files_dictionary[bkg][0], step):
      list_processes.append( ["python", "selectBranches.py", str(file_min), str(file_min+step), bkg, radius] )
      list_logfiles.append(log_folder+str(i)+"_log.txt")
      i += 1


print len(list_processes)

for i in list_processes:
    list_processes

# parallelise(list_processes, 20, list_logfiles)
