import numpy as np
import os.path
import os
import time
import sys

seed = 10
np.random.seed(seed)

branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_gen_jet = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]

def Sequential_var_selection(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
        if (not os.path.isfile(file_name)):
            continue
        jet_vars = np.load(file_name)
        jet_vars = jet_vars[(jet_vars[:,branch_names_jet.index("jetPt")]>pt_min)*(jet_vars[:,branch_names_jet.index("jetPt")]<pt_max)]
        try:
            jet_info = np.concatenate((jet_info,jet_vars))
        except:
            jet_info = jet_vars
        print jet_info.shape
    print jet_info.shape
    return jet_info

def merge_file(file_path, name_folder_output, name_variable, bkg, radius, file_min, file_max, pt_min, pt_max, step):
    first = 1
    for i in range(file_min, file_max, step):
        file_name = file_path+name_folder_output+"Sequential_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(i)+"_"+str(i+step)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        # print i, file_name
        if not os.path.isfile(file_name):
            F = open(file_path+name_folder_output+"/problem_"+bkg+radius+"_"+str(pt_min)+"_"+str(pt_max)+".txt","a")
            F.write("Missing "+bkg+"-"+radius+"-"+str(pt_min)+"-"+str(pt_max)+" "+str(i)+"\n")
            continue
        file = np.load(file_name)
        print file.shape
        if first:
            final = file
            first = 0
        else:
            final = np.concatenate((final,file))
        temp_fold = file_path+name_folder_output+"save/"
        if not os.path.exists(temp_fold):
            os.makedirs(temp_fold)
        os.system("mv "+file_name+" "+temp_fold)
    print final.shape
    np.save(file_path+name_folder_output+"Sequential_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", final)


##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = "/beegfs/desy/user/amalara/"
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = "input_varariables/NTuples_Tagger/Sequential/"
if not os.path.exists(file_path+name_folder_output):
    os.makedirs(file_path+name_folder_output)

file_min = int(sys.argv[1])
file_max = int(sys.argv[2])
pt_min = int(sys.argv[3])
pt_max = int(sys.argv[4])

name_variable = sys.argv[5]
bkg = sys.argv[6]
radius =sys.argv[7]
step =int(sys.argv[8])
flag_merge =int(sys.argv[9])

if name_variable=='norm':
    name_variable=""
print file_min, file_max, pt_min, pt_max, name_variable, bkg, radius, step, flag_merge

if flag_merge:
    merge_file(file_path, name_folder_output, name_variable, bkg, radius, file_min, file_max, pt_min, pt_max, step)
else:
    sequential = Sequential_var_selection(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
    np.save(file_path+name_folder_output+"Sequential_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", sequential)
