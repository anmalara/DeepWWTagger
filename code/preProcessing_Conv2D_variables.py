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

def matrix_norm(matrix):
    sum = 0
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            sum = matrix[x,y]*matrix[x,y]
    matrix = matrix / sum

def jet_preprocessing(cand_info, jet_info):
    translation = 1
    rotation = 1
    if translation:
        cand_info[:, branch_names_candidate.index("candEta"), :] = cand_info[:, branch_names_candidate.index("candEta"), :] - jet_info[:, [branch_names_jet.index("etaSub0")]]
        cand_info[:, branch_names_candidate.index("candPhi"), :] = cand_info[:, branch_names_candidate.index("candPhi"), :] - jet_info[:, [branch_names_jet.index("phiSub0")]]
    if rotation:
        pt_1 = jet_info[:, branch_names_jet.index("ptSub1")]
        eta_1 = jet_info[:, branch_names_jet.index("etaSub1")]
        phi_1 = jet_info[:, branch_names_jet.index("phiSub1")]
        px_1 = pt_1* np.cos(phi_1)
        py_1 = pt_1* np.sin(phi_1)
        pz_1 = pt_1* np.sinh(eta_1)
        theta_1 = np.arctan(py_1/pz_1) + math.pi/2.
        py_n = cand_info[:, branch_names_candidate.index("candPy"), :]
        pz_n = cand_info[:, branch_names_candidate.index("candPz"), :]
        cand_info[:, branch_names_candidate.index("candPy"), :] = py_n*np.cos(theta_1) - pz_n*np.sin(theta_1)
        cand_info[:, branch_names_candidate.index("candPz"), :] = py_n*np.sin(theta_1) - pz_n*np.cos(theta_1)

def jet_image_matrix(pf, eta_jet, phi_jet, radius):
    min_eta = -radius
    max_eta = +radius
    min_phi = -radius
    max_phi = +radius
    step_eta = 0.1
    step_phi = 0.1
    n_eta = int((max_eta-min_eta)/step_eta)
    n_phi = int((max_phi-min_phi)/step_phi)
    eta_block = np.zeros(n_eta)
    phi_block = np.zeros(n_phi)
    for x in range(0,n_eta,1):
        eta_block[x] = min_eta + x*step_eta
    for y in range(0,n_phi,1):
        phi_block[y] = min_phi + y*step_phi
    matrix = np.zeros((3,n_eta,n_phi))
    for i in range(pf.shape[1]):
        pt_pf = pf[branch_names_candidate.index("CandPt"),i]
        eta_pf = pf[branch_names_candidate.index("CandEta"),i] - eta_jet
        phi_pf = pf[branch_names_candidate.index("CandPhi"),i] - phi_jet
        pdgId_pf = int(abs(pf[branch_names_candidate.index("CandPdgId"),i]))
        if abs(eta_pf)>radius: continue
        if abs(phi_pf)>radius: continue
        found = 0
        x=0
        while (not found and x<n_eta-1):
            # print("not found and x = "+str(x))
            if eta_block[x] > eta_pf:
                found = 1
                x -= 1
                continue
            x += 1
        found = 0
        y=0
        while (not found and y<n_phi-1):
            # print("not found and y = "+str(y))
            if phi_block[y] > phi_pf:
                found = 1
                y -= 1
                continue
            y += 1
        if pdgId_pf == 130: #neutral hadron
                matrix[0,x,y] += 1
        elif pdgId_pf == 211: #charged hadron
                matrix[1,x,y] += 1
                matrix[2,x,y] += pt_pf
    return matrix


def create_image(jet_info, pf_info, radius):
    # print pf_info.shape
    for i in range(pf_info.shape[0]):
        # if (((i)*100./(pf_info.shape[0]))%10==0): print("progress: "+str(int((i)*100./(pf_info.shape[0])))+" %")
        eta_jet = jet_info[i,branch_names_jet.index("jetEta")]
        phi_jet = jet_info[i,branch_names_jet.index("jetPhi")]
        jet_image = jet_image_matrix(pf_info[i,:,:], eta_jet, phi_jet, radius)
        jet_image = jet_image.reshape((1,jet_image.shape[0],jet_image.shape[1],jet_image.shape[2]))
        if i==0:
            jet_images = jet_image
        else:
            jet_images = np.concatenate((jet_images,jet_image))
    # print("progress: 100%")
    return jet_images
        # print(jet_images.shape)
    # for a in range(jet_images.shape[1]):
    #     for b in range(jet_images.shape[2]):
    #         for c in range(jet_images.shape[3]):
    #             mean = np.mean(jet_images[:,a,b,c])
    #             std = np.std(jet_images[:,a,b,c])
    #             if(std!=0):
    #                 jet_images[:,a,b,c] = (jet_images[:,a,b,c] - mean)/std

def preprocessing_pt_order(jet_info, pf_info, pt_min, pt_max, isGen):
    if isGen:
        i = branch_names_gen_jet.index("GenJetPt")
    else:
        i = branch_names_jet.index("jetPt")
    jet_info_ = jet_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    pf_info_ = pf_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    return jet_info_, pf_info_

def jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    temp_time=time.time()
    radius_=0.8
    if "15" in radius:
        radius_=1.5
    # print radius_
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
        file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"cand_var_"+bkg+"_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        if name_variable=='gen_':
            isGen = 1
        else:
            isGen = 0
        jet_vars = np.load(file_name)
        pf_vars = np.load(file_name_1)
        jet_vars, pf_vars = preprocessing_pt_order(jet_vars, pf_vars, pt_min, pt_max, isGen)
        image = create_image(jet_vars, pf_vars, radius_)
        print image.shape
        np.save(file_path+name_folder_output+"JetImage_"+name_variable+"matrix_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", image)
    print("progress: 100%")
    print ("time needed: "+str((time.time()-temp_time))+" s")

def merge_file(file_path, name_folder_output, name_variable, bkg, radius, file_min, file_max, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        file_name = file_path+name_folder_output+"JetImage_"+name_variable+"matrix_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
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
    np.save(file_path+name_folder_output+"JetImage_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", final)

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = "/beegfs/desy/user/amalara/"
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = "input_varariables/NTuples_Tagger/JetImage/"
if not os.path.exists(file_path+name_folder_output):
    os.makedirs(file_path+name_folder_output)

file_min = int(sys.argv[1])
file_max = int(sys.argv[2])
pt_min = int(sys.argv[3])
pt_max = int(sys.argv[4])

name_variable = sys.argv[5]
bkg = sys.argv[6]
radius =sys.argv[7]
flag_merge =int(sys.argv[8])

if name_variable=='norm':
    name_variable=""

print file_min, file_max, pt_min, pt_max, name_variable, bkg, radius

if flag_merge:
    merge_file(file_path, name_folder_output, name_variable, bkg, radius, file_min, file_max, pt_min, pt_max)
else:
    jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
