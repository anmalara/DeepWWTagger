# from ROOT import *
from root_numpy import root2array
from root_numpy import rec2array
import numpy as np
import math
from math import *
import time
import sys
import os
import os.path

ncand = 40
selection_cuts = "((abs(jetEta)<=2.7)&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1))||(abs(jetEta)<2.4&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1)&&(jetChf>0)&&(jetChm>0)&&(jetElf<0.8))"

branch_name = ["lumi", "nvtx", "nJets", "jetBtag", "ncandidates", "nSubJets", "jetChf", "jetNhf", "jetMuf", "jetPhf", "jetElf"
               "jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag",
               "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3",
               "btagSub0", "btagSub1", "massSub0", "massSub1", "ptSub0", "ptSub1", "etaSub0", "etaSub1", "phiSub0", "phiSub1", "flavorSub0", "flavorSub1",
               "candEnergy", "candPx", "candPy", "candPz", "candEta", "candPhi", "candPt", "candMass"]

branch_names_gen = ["nGencandidates", "GenJetpt", "GenJeteta", "GenJetphi", "GenJetenergy", "GenJetmass", "isBJetGen", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4", "GenSoftDropMass", "GenCandPhi", "GenCandEta",
                    "GenCandPt", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandEnergy"]


branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_candidate = ["CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"]
branch_names_gen_jet = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]
branch_names_gen_cand = ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"]

def bubbleSort(matrix, col_max, px_index, py_index):
    for col in range(0, col_max-1):
        for i in range(col_max-1, col, -1):
            if math.sqrt(matrix[px_index][col]**2 + matrix[py_index][col]**2)< math.sqrt(matrix[px_index][i]**2 + matrix[py_index][i]**2):
                matrix[:,[col, i]] = matrix[:,[i, col]]

def selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen):
    file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
    file = rec2array(file)
    if isGen:
        px_index = branch_names.index("GenCandPx")
        py_index = branch_names.index("GenCandPy")
    else:
        px_index = branch_names.index("CandPx")
        py_index = branch_names.index("CandPy")
    print("len_file = "+str(len(file)))
    for x in range(0,len(file)):
        for y in range(0,len(branch_names)):
            file[x,y].resize(ncand, refcheck=False)
            temp = file[x,y].reshape(1,ncand)
            if y == 0:
                info = temp
            else:
                info = np.concatenate((info, temp))
        bubbleSort(info,ncand, px_index, py_index)
        temp_jet = info
        temp_jet=temp_jet.reshape(1,len(branch_names), ncand)
        if x==0:
            info_candidates = temp_jet
        else:
            info_candidates = np.concatenate((info_candidates, temp_jet))
        if (x-1)%1000 == 0:
            print("info_candidates.shape = "+str(info_candidates.shape))
    print("done")
    return info_candidates


def selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts):
    file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
    file = rec2array(file)
    for x in range(file.shape[0]):
        for y in range(file.shape[1]):
            if file[x,y].shape[0]<1:
                # if "Gen" not in branch_names[y]:
                #     print(branch_names[y])
                continue
            file[x,y] = file[x,y][0]
    return file


def selectBranches(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius):
    temp_time=time.time()
    for i in range(file_min, file_max):
        file_name = file_path+name_folder+"0000/"+name_variable+str(i)+".root"
        if i >= 1000:
            file_name = file_path+name_folder+"0001/"+name_variable+str(i)+".root"
        if not os.path.isfile(file_name):
            continue
        tree_name = "boosted"+radius+"/events"
        isGen = 0
        branch_names = branch_names_jet
        jet_info = selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts)
        outputfile = file_path+name_folder_output+radius+"/jet_var_"+bkg+"_"+str(i)
        np.save( outputfile+".npy",jet_info)
        branch_names = branch_names_candidate
        pf_info = selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen)
        outputfile = file_path+name_folder_output+radius+"/cand_var_"+bkg+"_"+str(i)
        np.save( outputfile+".npy",pf_info)
        isGen = 1
        branch_names = branch_names_gen_jet
        gen_jet_info = selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts)
        outputfile = file_path+name_folder_output+radius+"/gen_jet_var_"+bkg+"_"+str(i)
        np.save( outputfile+".npy",gen_jet_info)
        branch_names = branch_names_gen_cand
        gen_pf_info = selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen)
        outputfile = file_path+name_folder_output+radius+"/gen_cand_var_"+bkg+"_"+str(i)
        np.save( outputfile+".npy",gen_pf_info)
        print jet_info.shape, pf_info.shape, gen_jet_info.shape, gen_pf_info.shape
        if jet_info.shape[0]!=gen_jet_info.shape[0]:
            if i==file_min:
                F = open(file_path+name_folder_output+radius+"/problem.txt","w")
            else:
                F = open(file_path+name_folder_output+radius+"/problem.txt","a")
            F.write("Check in "+bkg+" "+str(i))
            F.close()
    print ("time needed: "+str((time.time()-temp_time))+" s")




##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = "/beegfs/desy/user/amalara/"
file_min = int(sys.argv[1])
file_max = int(sys.argv[2])
bkg = sys.argv[3]
radius = sys.argv[4]
if bkg=="Higgs":
    name_folder = "Ntuples/MC-CandidatesP8-Higgs-GenInfos-v4/180413_072857/"
if bkg=="QCD":
    name_folder = "Ntuples/MC-CandidatesP8-QCD-GenInfos-v4/180413_073311/"

name_variable = "flatTreeFile"+bkg+"-nano_"
name_folder_output = "input_varariables/NTuples_Tagger/"+bkg+"_"

print name_folder, name_variable, name_folder_output, file_min, file_max, bkg, radius

if not os.path.exists(file_path+name_folder_output+radius):
    os.makedirs(file_path+name_folder_output+radius)
print("\n")
print("process : Higgs_"+radius)
selectBranches(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius)
