import numpy as np
import time
import os
import os.path
import sys

from root_numpy import *
import math
from math import *

from variables import *

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
        cand_info[:, branch_names_dict["CandInfo"].index("CandEta"), :] = cand_info[:, branch_names_dict["CandInfo"].index("CandEta"), :] - jet_info[:, [branch_names_dict["JetInfo"].index("etaSub0")]]
        cand_info[:, branch_names_dict["CandInfo"].index("CandPhi"), :] = cand_info[:, branch_names_dict["CandInfo"].index("CandPhi"), :] - jet_info[:, [branch_names_dict["JetInfo"].index("phiSub0")]]
    if rotation:
        pt_1 = jet_info[:, branch_names_dict["JetInfo"].index("ptSub1")]
        eta_1 = jet_info[:, branch_names_dict["JetInfo"].index("etaSub1")]
        phi_1 = jet_info[:, branch_names_dict["JetInfo"].index("phiSub1")]
        px_1 = pt_1* np.cos(phi_1)
        py_1 = pt_1* np.sin(phi_1)
        pz_1 = pt_1* np.sinh(eta_1)
        theta_1 = np.arctan(py_1/pz_1) + math.pi/2.
        py_n = cand_info[:, branch_names_dict["CandInfo"].index("candPy"), :]
        pz_n = cand_info[:, branch_names_dict["CandInfo"].index("candPz"), :]
        cand_info[:, branch_names_dict["CandInfo"].index("candPy"), :] = py_n*np.cos(theta_1) - pz_n*np.sin(theta_1)
        cand_info[:, branch_names_dict["CandInfo"].index("candPz"), :] = py_n*np.sin(theta_1) - pz_n*np.cos(theta_1)

def jet_image_matrix(pf, eta_jet, phi_jet, radius):
    min_eta = -radius
    max_eta = +radius
    min_phi = -radius
    max_phi = +radius
    n_eta = int((max_eta-min_eta)/step_eta)
    n_phi = int((max_phi-min_phi)/step_phi)
    eta_block = np.zeros(n_eta)
    phi_block = np.zeros(n_phi)
    #created the minimum edges on the 2 axes
    for x in range(0,n_eta,1):
        eta_block[x] = min_eta + x*step_eta
    for y in range(0,n_phi,1):
        phi_block[y] = min_phi + y*step_phi
    matrix = np.zeros((n_images,n_eta,n_phi))
    #for each pf cand (n_cand)
    for i in range(pf.shape[1]):
        pt_pf = pf[branch_names_dict["CandInfo"].index("CandPt"),i]
        eta_pf = pf[branch_names_dict["CandInfo"].index("CandEta"),i] - eta_jet
        phi_pf = pf[branch_names_dict["CandInfo"].index("CandPhi"),i] - phi_jet
        pdgId_pf = int(abs(pf[branch_names_dict["CandInfo"].index("CandPdgId"),i]))
        if abs(eta_pf)>radius:
            continue
        if abs(phi_pf)>radius:
            continue
        found = 0
        x=0
        # find the right bin in eta-phi matrix
        while (not found and x<n_eta-1):
            if eta_block[x] > eta_pf:
                found = 1
                x -= 1
                continue
            x += 1
        found = 0
        y=0
        while (not found and y<n_phi-1):
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
    for i in range(pf_info.shape[0]):
        eta_jet = jet_info[i,branch_names_dict["JetInfo"].index("jetEta")]
        phi_jet = jet_info[i,branch_names_dict["JetInfo"].index("jetPhi")]
        jet_image = jet_image_matrix(pf_info[i,:,:], eta_jet, phi_jet, radius)
        jet_image = jet_image.reshape((1,jet_image.shape[0],jet_image.shape[1],jet_image.shape[2]))
        if i==0:
            jet_images = jet_image
        else:
            jet_images = np.concatenate((jet_images,jet_image))
    # numpy.ndarray (n_events, n_images, 2*radius, 2*radius )
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
        i = branch_names_dict["GenJetInfo"].index("GenJetPt")
    else:
        i = branch_names_dict["JetInfo"].index("jetPt")
    jet_info_ = jet_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    pf_info_  =  pf_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
    # selects events in n_events according to pt cuts: Same selection applied to jet and pf-cands
    return jet_info_, pf_info_

def jetImage_inputFiles(file_path, name_folder, info, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    radius_= 0.8
    if "15" in radius:
        radius_ = 1.5
    for i in range(file_min, file_max):
        file_name   = file_path+name_folder+bkg+"_"+radius+"/"+info[0]+"_"+bkg+"_"+str(i)+".npy"
        file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+info[1]+"_"+bkg+"_"+str(i)+".npy"
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        isGen = 0
        if "Gen" in info[0]:
            isGen = 1
        jet_vars = np.load(file_name)
        # numpy.ndarray (n_events,branches.size())
        pf_vars = np.load(file_name_1)
        # numpy.ndarray (n_events,branches.size(), ncand)
        jet_vars, pf_vars = preprocessing_pt_order(jet_vars, pf_vars, pt_min, pt_max, isGen)
        if len(jet_vars) == 0 or len(pf_vars) == 0:
            print i
            continue
        #same shape, with less n_events
        image = create_image(jet_vars, pf_vars, radius_)
        np.save(file_path+name_folder_output+"JetImage_"+info[0]+"_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", image)

def merge_file(file_path, name_folder_output, info, bkg, radius, file_min, file_max, pt_min, pt_max):
    first = True
    for i in range(file_min, file_max+1):
        file_name = file_path+name_folder_output+"JetImage_"+info[0]+"_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        if not os.path.isfile(file_name):
            with open(file_path+name_folder_output+"/problem_"+bkg+radius+"_"+str(pt_min)+"_"+str(pt_max)+".txt","a") as F:
                F.write("Missing "+bkg+"-"+radius+"-"+str(pt_min)+"-"+str(pt_max)+" "+str(i)+"\n")
            continue
        file = np.load(file_name)
        # print file.shape
        if first:
            final = file
            first = False
        else:
            final = np.concatenate((final,file))
        print final.shape, "   ", (final.size * final.itemsize)/1000000000., " Gb"
        temp_fold = file_path+name_folder_output+"save/"
        if not os.path.exists(temp_fold):
            os.makedirs(temp_fold)
        os.system("mv "+file_name+" "+temp_fold)
        file = []
    if first == 0:
        print final.shape
        np.save(file_path+name_folder_output+"JetImage_"+info[0]+"_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", final)

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = out_path
name_folder = "input_varariables/NTuples_Tagger/"

try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    bkg = sys.argv[3]
    radius = sys.argv[4]
    info = [sys.argv[5],sys.argv[6]]
    pt_min = int(sys.argv[7])
    pt_max = int(sys.argv[8])
    flag_merge =int(sys.argv[9])
except Exception as e:
    file_min = 0
    file_max = 10
    bkg = "Higgs"
    radius = "AK8"
    info = ["JetInfo","CandInfo"]
    pt_min = 300
    pt_max = 500
    flag_merge = 0

# if info=='norm':
#     info=""

name_folder_output = name_folder+"JetImage/"+bkg+"_"+radius+"/"
if not os.path.exists(file_path+name_folder_output):
    os.makedirs(file_path+name_folder_output)

if flag_merge:
    merge_file(file_path, name_folder_output, info, bkg, radius, file_min, file_max, pt_min, pt_max)
else:
    jetImage_inputFiles(file_path, name_folder, info, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
