from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time

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



radius = 0.8
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
eta_block
matrix.shape

def jet_image_matrix(pf, eta_jet, phi_jet):
    matrix = np.zeros((3,n_eta,n_phi))
    for i in range(pf.shape[1]):
        # print(str(i)+" of "+str(pf.shape[1])+" jet_image_matrix")
        pt_pf = pf[branch_names_candidate.index("candPt"),i]
        eta_pf = pf[branch_names_candidate.index("candEta"),i] - eta_jet
        phi_pf = pf[branch_names_candidate.index("candPhi"),i] - phi_jet
        pdgId_pf = abs(pf[branch_names_candidate.index("candPdgId"),i])
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


def create_image(jet_info, pf_info):
    for i in range(pf_info.shape[0]):
        # if (int((i)*100./(pf_info.shape[0]))%10==0): print("progress: "+str(int((i)*100./(pf_info.shape[0])))+" %")
        # if i%1000==0: print(str(i)+" of "+str(pf_info.shape[0])+" create_image")
        eta_jet = jet_info[i,branch_names_jet.index("jetEta")]
        phi_jet = jet_info[i,branch_names_jet.index("jetPhi")]
        jet_image = jet_image_matrix(pf_info[i,:,:], eta_jet, phi_jet)
        jet_image = jet_image.reshape((1,jet_image.shape[0],jet_image.shape[1],jet_image.shape[2]))
        if i==0:
            jet_images = jet_image
        else:
            jet_images = np.concatenate((jet_images,jet_image))
        # print(jet_images.shape)
    for a in range(jet_images.shape[1]):
        for b in range(jet_images.shape[2]):
            for c in range(jet_images.shape[3]):
                mean = np.mean(jet_images[:,a,b,c])
                std = np.std(jet_images[:,a,b,c])
                if(std!=0):
                    jet_images[:,a,b,c] = (jet_images[:,a,b,c] - mean)/std
    print(jet_images.shape)
    return jet_images

def preprocessing_pt_order(jet_info, pf_info, pt_min, pt_max, isGen):
    if isGen:
        i = branch_names_gen_jet.index("GenJetPt")
    else:
        i = branch_names_jet.index("jetPt")
    jet_info_ = jet_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    pf_info_ = pf_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    return jet_info_, pf_info_

def jetImage_inputFiles(file_min, file_max, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        if (int((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+"QCD/LoLa_jet_QCD_"+str(i)+".npy"
        file_name_1 = file_path+"QCD/LoLa_cand_QCD_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        jet_vars_QCD = np.load(file_name)
        pf_vars_QCD = np.load(file_name_1)
        jet_vars_QCD, pf_vars_QCD = preprocessing_pt_order(jet_vars_QCD, pf_vars_QCD, pt_min, pt_max)
        if first:
            first = 0
            jet_info_QCD = jet_vars_QCD
            pf_info_QCD = pf_vars_QCD
        else:
            jet_info_QCD = np.concatenate((jet_info_QCD,jet_vars_QCD))
            pf_info_QCD = np.concatenate((pf_info_QCD,pf_vars_QCD))
    if not first:
        QCD = create_image(jet_info_QCD, pf_info_QCD)
        np.save("file/QCD_image_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", QCD)



def jetImage_QCD_inputFiles(file_min, file_max, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        if (int((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+"QCD/LoLa_jet_QCD_"+str(i)+".npy"
        file_name_1 = file_path+"QCD/LoLa_cand_QCD_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        jet_vars_QCD = np.load(file_name)
        pf_vars_QCD = np.load(file_name_1)
        jet_vars_QCD, pf_vars_QCD = preprocessing_pt_order(jet_vars_QCD, pf_vars_QCD, pt_min, pt_max)
        if first:
            first = 0
            jet_info_QCD = jet_vars_QCD
            pf_info_QCD = pf_vars_QCD
        else:
            jet_info_QCD = np.concatenate((jet_info_QCD,jet_vars_QCD))
            pf_info_QCD = np.concatenate((pf_info_QCD,pf_vars_QCD))
    if not first:
        QCD = create_image(jet_info_QCD, pf_info_QCD)
        np.save("file/QCD_image_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", QCD)


def jetImage_Higgs_inputFiles(file_min, file_max, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        if (int((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+"Higgs/LoLa_jet_Higgs_"+str(i)+".npy"
        file_name_1 = file_path+"Higgs/LoLa_cand_Higgs_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        jet_vars_higgs = np.load(file_name)
        pf_vars_higgs = np.load(file_name_1)
        jet_vars_higgs, pf_vars_higgs = preprocessing_pt_order(jet_vars_higgs, pf_vars_higgs, pt_min, pt_max)
        if first:
            first = 0
            jet_info_higgs = jet_vars_higgs
            pf_info_higgs = pf_vars_higgs
        else:
            jet_info_higgs = np.concatenate((jet_info_higgs,jet_vars_higgs))
            pf_info_higgs = np.concatenate((pf_info_higgs,pf_vars_higgs))
    if not first:
        Higgs = create_image(jet_info_higgs, pf_info_higgs)
        np.save("file/Higgs_image_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", Higgs)

def lola_Higgs_inputFiles(file_min, file_max, pt_min, pt_max):
   first = 1
   for i in range(file_min, file_max):
      if (int((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
      file_name = file_path+"Higgs/LoLa_jet_Higgs_"+str(i)+".npy"
      file_name_1 = file_path+"Higgs/LoLa_cand_Higgs_"+str(i)+".npy"
      if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
         continue
      jet_vars_higgs = np.load(file_name)
      pf_vars_higgs = np.load(file_name_1)
      jet_vars_higgs, pf_vars_higgs = preprocessing_pt_order(jet_vars_higgs, pf_vars_higgs, pt_min, pt_max)
      if first:
         first = 0
         pf_info_higgs = pf_vars_higgs
      else:
         pf_info_higgs = np.concatenate((pf_info_higgs,pf_vars_higgs))
   if not first:
      print pf_info_higgs.shape
      Higgs = pf_info_higgs[:,(branch_names_candidate.index("candEnergy"), branch_names_candidate.index("candPx"), branch_names_candidate.index("candPy"), branch_names_candidate.index("candPz")),:]
      print Higgs.shape
      np.save("file/Higgs_lola_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", Higgs)



def lola_QCD_inputFiles(file_min, file_max, pt_min, pt_max):
   first = 1
   for i in range(file_min, file_max):
      if (int((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
      file_name = file_path+"QCD/LoLa_jet_QCD_"+str(i)+".npy"
      file_name_1 = file_path+"QCD/LoLa_cand_QCD_"+str(i)+".npy"
      if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
         continue
      jet_vars_QCD = np.load(file_name)
      pf_vars_QCD = np.load(file_name_1)
      jet_vars_QCD, pf_vars_QCD = preprocessing_pt_order(jet_vars_QCD, pf_vars_QCD, pt_min, pt_max)
      if first:
         first = 0
         pf_info_QCD = pf_vars_QCD
      else:
         pf_info_QCD = np.concatenate((pf_info_QCD,pf_vars_QCD))
   if not first:
      print pf_info_QCD.shape
      QCD = pf_info_QCD[:,(branch_names_candidate.index("candEnergy"), branch_names_candidate.index("candPx"), branch_names_candidate.index("candPy"), branch_names_candidate.index("candPz")),:]
      print QCD.shape
      np.save("file/QCD_lola_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", QCD)

def merge_file(file_min, file_max, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max, 20):
        file_name = "file/QCD_image_preProcess_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        temp_name = "file/QCD_image_preProcess_file_"+str(i)+"_"+str(i+20)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        if not os.path.isfile(temp_name):
            continue
        temp = np.load(temp_name)
        if first:
            first = 0
            if not os.path.isfile(file_name):
                final = temp
            else:
                final = np.load(file_name)
                print final.shape
        final = np.concatenate((final,temp))
        temp_fold = "file/save/"
        os.system("mv "+temp_name+" "+temp_fold)
    if not first:
        print final.shape
        np.save(file_name, final)


def lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    first = 1
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
        file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"cand_var_"+bkg+"_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            # print "problem at "+str(i)
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


# file_min = int(sys.argv[1])
# file_max = int(sys.argv[2])
file_min = 0
file_max = 10
pt_min = 300
pt_max = 500

name_variable = "gen_"
# name_variable = ""
bkg = "QCD"

radius = "AK8"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
radius = "AK15"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
radius = "CA15"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)

bkg = "Higgs"

radius = "AK8"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
radius = "AK15"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
radius = "CA15"
lola_input_variable(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)


# jetImage_Higgs_inputFiles(0,10,300,500)
# jetImage_Higgs_inputFiles(0,10,500,10000)
#
# temp_time=time.time()
# for i in range(600, 1001, 20):
#     print("range "+str(i)+" "+str(i+20))
#     jetImage_QCD_inputFiles(i,i+20,300,500)
#     jetImage_QCD_inputFiles(i,i+20,500,10000)
#     print ((time.time()-temp_time)/60.)
#     temp_time=time.time()

# merge_file(0, 2000, 300, 500)
# merge_file(0, 2000, 500, 10000)

# lola_Higgs_inputFiles(0,10,300,500)
# lola_Higgs_inputFiles(0,10,500,10000)
#
# temp_time=time.time()
# for i in range(600, 2001, 1000):
#     print("range "+str(i)+" "+str(i+20))
#     lola_QCD_inputFiles(i,i+20,300,500)
#     lola_QCD_inputFiles(i,i+20,500,10000)
#     print ((time.time()-temp_time)/60.)
#     temp_time=time.time()
