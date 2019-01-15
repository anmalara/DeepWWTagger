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


log_folder= "./log_CreateInputs/"
for file in glob(log_folder+"*txt"):
    os.remove(file)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

list_batch      = []
list_processes  = []
list_logfiles   = []

i = 0
for bkg in bkgs:
    if not "WJets" in bkg:
        continue
    for radius in radii:
        if not "AK15" in radius:
            continue
        for pt_index, pt_min in enumerate(pts_min):
            if pt_min == 300:
                continue
            for index in range(0,files_dictionary[bkg]["maximum"],step):
                list_batch.append( ["condor_submit", "condor_CreateInputs.submit", "-a", "filemin="+str(index), "filemax="+str(index+step), "sample="+bkg, "radius="+radius, "ptmin="+str(pt_min), "ptmax="+str(pts_max[pt_index]), "myProcess="+str(i), "-queue", "1" ])
                list_processes.append( ["python", "CreateInputs.py", str(index), str(index+step), bkg, radius, str(pt_min), str(pts_max[pt_index])])
                list_logfiles.append(log_folder+str(i)+"_log.txt")
                i += 1


# print len(list_batch)
# for i in list_batch:
#     print i

print len(list_processes)
for i in list_processes:
    print i

# parallelise(list_batch, 20)
parallelise(list_processes, 40, list_logfiles)
