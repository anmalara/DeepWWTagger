from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time
import sys

seed = 7
np.random.seed(seed)
start_time=time.time()
temp_time = start_time


ncand = 40
selection_cuts = "((abs(jetEta)<=2.7)&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1))||(abs(jetEta)<2.4&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1)&&(jetChf>0)&&(jetChm>0)&&(jetElf<0.8))"

branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_candidate = ["CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"]
branch_names_gen_jet = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]
branch_names_gen_cand = ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"]

def preprocessing_pt_order(jet_info, pf_info, pt_min, pt_max, isGen):
    if isGen:
        i = branch_names_gen_jet.index("GenJetPt")
    else:
        i = branch_names_jet.index("jetPt")
    jet_info_ = jet_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    pf_info_ = pf_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    return jet_info_, pf_info_

def lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
        file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"cand_var_"+bkg+"_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            # print "problem at "+file_name
            continue
        jet_vars = np.load(file_name)
        cand_vars = np.load(file_name_1)
        if name_variable=='gen_':
            isGen = 1
        else:
            isGen = 0
        jet_vars, cand_vars = preprocessing_pt_order(jet_vars, cand_vars, pt_min, pt_max, isGen)
        if isGen:
            cand_vars = cand_vars[:,(branch_names_gen_cand.index("GenCandEnergy"), branch_names_gen_cand.index("GenCandPx"), branch_names_gen_cand.index("GenCandPy"), branch_names_gen_cand.index("GenCandPz")),:]
        else:
            cand_vars = cand_vars[:,(branch_names_candidate.index("CandEnergy"), branch_names_candidate.index("CandPx"), branch_names_candidate.index("CandPy"), branch_names_candidate.index("CandPz")),:]
        if first:
            first = 0
            Lola_input = cand_vars
        else:
            Lola_input = np.concatenate((Lola_input,cand_vars))
    if not first:
        print Lola_input.shape
        np.save(file_path+name_folder_output+"lola_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", Lola_input)

def merge_file(file_path, name_folder_output, name_variable, bkg, radius, file_min, file_max, pt_min, pt_max, step):
    first = 1
    for i in range(file_min, file_max, step):
        file_name = file_path+name_folder_output+"lola_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(i)+"_"+str(i+step)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
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
    np.save(file_path+name_folder_output+"lola_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", final)

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = "/beegfs/desy/user/amalara/"
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = "input_varariables/NTuples_Tagger/Lola/"
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
    lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
