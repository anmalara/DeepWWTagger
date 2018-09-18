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
    step = 1

pts_min = [300,500]
pts_max = [500,10000]

log_folder= "./log_preProcessing_Conv/"
filelist=glob.glob(log_folder+"*txt")

list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []

for file_min in range(min, max, step):
    for bkg in ["Higgs"]:
        for radius in ["AK8"]:
            for name_variable in ["norm"]:
                for index, pt_min in enumerate(pts_min):
                  list1.append(str(file_min))
                  list2.append(bkg)
                  list3.append(radius)
                  list4.append(name_variable)
                  list5.append(str(pt_min))
                  list6.append(str(pts_max[index]))

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
    for info1 in branch_names_dict:
      if "Event" in info1 or "Extra" in info1 or "Sub" in info1 or "Gen" in info1:
        continue
      for info2 in branch_names_dict:
        if "Event" in info2 or "Extra" in info2 or "Sub" in info2 or "Gen" in info2:
          continue
        if info1==info2:
          continue
        if "Cand" in info1 and "Jet" in info2:
          continue
        for index, pt_min in enumerate(pts_min):
          for file_min in range(min, files_dictionary[bkg][0], step):
            list_processes.append( ["python", "preProcessing_Conv2D_variables.py", str(file_min), str(file_min+step), bkg, radius, info1, info2, str(pt_min), str(pts_max[index]), str(0)])
            list_logfiles.append(log_folder+str(i)+"_log.txt")
            i += 1
          list_processes_merge.append( ["python", "preProcessing_Conv2D_variables.py", str(min), str(files_dictionary[bkg][0]), bkg, radius, info1, info2, str(pt_min), str(pts_max[index]), str(step), str(1)])
          list_logfiles_merge.append(log_folder+str(j)+"_log.txt")
          j += 1

print len(list_processes)

parallelise(list_processes, 20, list_logfiles)

####################
# Now Merge
####################

print len(list_processes_merge)
# parallelise(list_processes_merge, 5, list_logfiles_merge)
