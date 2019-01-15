from variables import *
from glob import glob

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from parallelise import *


@timeit
def CountFiles():
    list_batch = []
    list_processes = []
    list_logfiles = []
    log_folder= "./log_selectBranches/"
    i_ =0
    for bkg in bkgs:
        for radius in radii:
            found = False
            for info in branch_names_dict:
                outputdir = out_path+"input_varariables/NTuples_Tagger/"+bkg+"_"+radius+"/"+info+"/"
                count = len(files_dictionary[bkg]["elements"])
                for file_ in glob(outputdir+"*.npy"):
                    count -= 1
                if count ==0:
                    # print "OK ", outputdir
                    pass
                else:
                    print count,"/", len(files_dictionary[bkg]["elements"]), outputdir
                    if found:
                        continue
                    if (len(files_dictionary[bkg]["elements"])-count) != 0:
                        file_ = file_[:file_.rfind("_")+1]
                    else:
                        file_ = outputdir+info+"_"
                    for file_index in files_dictionary[bkg]["elements"]:
                        if not os.path.exists(file_+str(file_index)+".npy"):
                            found = True
                            list_batch.append( ["condor_submit", "condor_selectBranches.submit", "-a", "filemin="+str(file_index), "filemax="+str(file_index+1), "sample="+bkg, "radius="+radius, "myProcess="+str(file_index), "-queue", "1" ])
                            list_processes.append( ["python", "selectBranches.py", str(file_index), str(file_index+1), bkg, radius] )
                            list_logfiles.append(log_folder+str(i_)+"_log_reload.txt")
                            i_+=1
    print len(list_processes)
    # for i in list_processes:
    #     print i
    # parallelise(list_batch, 20)
    # parallelise(list_processes, 20, list_logfiles)


CountFiles()


def CountEvents(bkg,outputdir):
    count = len(files_dictionary[bkg]["elements"])
    for file in glob(outputdir):
        count -= 1
    return count


@timeit
def CountInputs():
    list_batch = []
    list_processes = []
    list_logfiles = []
    log_folder= "./log_CreateInputs/"
    i_ =0
    outputdir = out_path+"input_varariables/NTuples_Tagger/Inputs/"
    for bkg in bkgs:
        for radius in radii:
            for pt_index, pt_min in enumerate(pts_min):
                pt_max = pts_max[pt_index]
                found = False
                for method in ["JetImage", "JetVariables"]+branch_names_dict.keys():
                    file_name = bkg+"_"+radius+"/"+method
                    file_name += "/file_*_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
                    count = CountEvents(bkg,outputdir+file_name)
                    if count ==0:
                        # print "OK ", file_name
                        pass
                    else:
                        if found:
                            continue
                        print count,"/", len(files_dictionary[bkg]["elements"]), file_name
                        for file_index in files_dictionary[bkg]["elements"]:
                            file_name = bkg+"_"+radius+"/"+method
                            file_name += "/file_"+str(file_index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
                            if not os.path.exists(outputdir+file_name):
                                found = True
                                list_batch.append( ["condor_submit", "condor_CreateInputs.submit", "-a", "filemin="+str(file_index), "filemax="+str(file_index+1), "sample="+bkg, "radius="+radius, "ptmin="+str(pt_min), "ptmax="+str(pt_max), "myProcess="+str(i_), "-queue", "1" ])
                                list_processes.append( ["python", "CreateInputs.py", str(file_index), str(file_index+1), bkg, radius, str(pt_min), str(pt_max)])
                                list_logfiles.append(log_folder+str(i_)+"_log_reload.txt")
                                i_+=1
    print len(list_processes)
    for i in list_processes:
        print i
    # parallelise(list_batch, 20)
    parallelise(list_processes, 20, list_logfiles)


CountInputs()
