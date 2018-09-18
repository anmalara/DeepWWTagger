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
  step = 100

pts_min = [300,500]
pts_max = [500,10000]

log_folder= "./log_preProcessing_Sequential/"
filelist=glob.glob(log_folder+"*txt")
for file in filelist:
  os.remove(file)

if not os.path.exists(log_folder):
  os.makedirs(log_folder)

list_processes        = []
list_logfiles         = []
list_processes_merge  = []
list_logfiles_merge   = []
i = 0
j = 0
for bkg in bkgs:
  for radius in radii:
    for info in branch_names_dict:
      if "Event" in info or "Extra" in info or "Cand" in info:
        continue
      for index, pt_min in enumerate(pts_min):
        for file_min in range(min, files_dictionary[bkg][0], step):
          list_processes.append( ["python", "preProcessing_Sequential_variables.py", str(file_min), str(file_min+step), bkg, radius, info, str(pt_min), str(pts_max[index]), str(step), str(0)])
          list_logfiles.append(log_folder+str(i)+"_log.txt")
          i += 1
        list_processes_merge.append( ["python", "preProcessing_Sequential_variables.py", str(min), str(files_dictionary[bkg][0]), bkg, radius, info, str(pt_min), str(pts_max[index]), str(step), str(1)])
        list_logfiles_merge.append(log_folder+str(j)+"_log.txt")
        j += 1

print len(list_processes)
parallelise(list_processes, 20, list_logfiles)

####################
# Now Merge
####################

print len(list_processes_merge)
parallelise(list_processes_merge, 5, list_logfiles_merge)
