import sys
import os
import os.path
import glob
import time
import copy
import numpy as np
from ROOT import TFile, TCanvas, TLegend, TH1F, TH2F, TColor, TAxis
from ROOT import kWhite, kBlack, kGray, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan, kOrange, kSpring, kTeal, kAzure, kViolet, kPink
from ROOT import kNone, gStyle
from root_numpy import *
import math
from math import *
from math import pi as PI

from variables import *

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from tdrstyle_all import *


colors = [kBlack, kRed+1, kBlue-4, kGreen-2, kOrange, kMagenta, kViolet-3, kCyan, kSpring, kTeal, kYellow+1, kPink+10, kAzure+7, kAzure+1, kRed+3, kGray]

jet_eta_index = branch_names_dict["JetInfo"].index("jetEta")
jet_phi_index = branch_names_dict["JetInfo"].index("jetPhi")
jet_pt_index = branch_names_dict["JetInfo"].index("jetPt")
genjet_pt_index = branch_names_dict["GenJetInfo"].index("GenJetPt")

cand_eta_index = branch_names_dict["CandInfo"].index("CandEta")
cand_phi_index = branch_names_dict["CandInfo"].index("CandPhi")
cand_pt_index = branch_names_dict["CandInfo"].index("CandPt")
cand_pdgID_index = branch_names_dict["CandInfo"].index("CandPdgId")

subjet0_eta_index = branch_names_dict["SubJetInfo"].index("etaSub0")
subjet0_phi_index = branch_names_dict["SubJetInfo"].index("phiSub0")
subjet1_eta_index = branch_names_dict["SubJetInfo"].index("etaSub1")
subjet1_phi_index = branch_names_dict["SubJetInfo"].index("phiSub1")


@timeit
def plotJetImages(array, process, output_path="./",):
    if "15" in process:
        radius_= 1.5
    elif "8" in process:
        radius_= 0.8
    print process, radius_
    Nbins = int(2*radius_/step_eta)
    canvases = []
    histos = []
    for n in range(n_images):
        c = tdrCanvas(process+"_"+images_dict[n], -radius_, radius_, -radius_, radius_, "#eta", "#phi", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
        gStyle.SetOptStat(0)
        c.SetRightMargin(1)
        c.SetLogz(1)
        h = TH2F( process+"_"+images_dict[n], process+"_"+images_dict[n], Nbins, -radius_, radius_, Nbins, -radius_, radius_)
        for i in range(1,Nbins):
            for j in range(1,Nbins):
                h.SetBinContent(i,j,array[n,i,j])
        h.Draw("colz")
        histos.append(h)
        canvases.append(c)
    return canvases, histos


def jet_preprocessing(pf_vars, jet_vars, subjet_vars):
    eta = pf_vars[:, cand_eta_index, :]
    phi = pf_vars[:, cand_phi_index, :]
    # eta_subjet0 = subjet_vars[:, [subjet0_eta_index]].astype(variable_type)
    # phi_subjet0 = subjet_vars[:, [subjet0_phi_index]].astype(variable_type)
    if translation:
        eta -= eta[:,[0]]
        phi -= phi[:,[0]]
        # eta -= eta_subjet0
        # phi -= phi_subjet0
    if rotation:
        eta_2 = eta[:,1].astype(variable_type)
        phi_2 = phi[:,1].astype(variable_type)
        # eta_2 = subjet_vars[:, subjet1_eta_index].astype(variable_type)
        # phi_2 = subjet_vars[:, subjet1_phi_index].astype(variable_type)
        alpha = np.arctan2(phi_2,eta_2)
        alpha[alpha<0] += 2*PI
        c = np.cos(alpha)
        s = np.sin(alpha)
        angles = np.array((eta,phi)).swapaxes(0,1).swapaxes(1,2)
        R = np.array(((c,-s), (s, c))).swapaxes(0,1).swapaxes(0,2)
        angles = np.matmul(angles, R)
        pf_vars[:, cand_eta_index, :] = angles[:,:,0]
        pf_vars[:, cand_phi_index, :] = angles[:,:,1]


def jet_preprocessing_old(pf_vars, jet_vars, subjet_vars):
    if translation:
        pf_vars[:, cand_eta_index, :] = np.subtract(pf_vars[:, cand_eta_index, :], subjet_vars[:, [subjet0_eta_index] ])
        pf_vars[:, cand_phi_index, :] = np.subtract(pf_vars[:, cand_phi_index, :], subjet_vars[:, [subjet0_phi_index] ])
        subjet_vars[:, [subjet1_eta_index]] = np.subtract(subjet_vars[:, [subjet1_eta_index]], subjet_vars[:, [subjet0_eta_index] ])
        subjet_vars[:, [subjet1_phi_index]] = np.subtract(subjet_vars[:, [subjet1_phi_index]], subjet_vars[:, [subjet0_phi_index] ])
        # pf_vars[:, cand_eta_index, :] = np.subtract(pf_vars[:, cand_eta_index, :], pf_vars[:, cand_pt_index, :].max(axis=1) )
        # pf_vars[:, cand_phi_index, :] = np.subtract(pf_vars[:, cand_eta_index, :], pf_vars[:, cand_pt_index, :].max(axis=1) )
    if rotation:
      eta = pf_vars[:, cand_eta_index, :]
      phi = pf_vars[:, cand_phi_index, :]
      eta_2 = subjet_vars[:, subjet1_eta_index]
      phi_2 = subjet_vars[:, subjet1_phi_index]
      for i in range(eta_2.shape[0]):
          if eta_2[i]==0:
              eta_2[i] = 0.000000001
          if np.abs(eta_2[i])>PI:
              eta_2[i] = PI
      for i in range(phi_2.shape[0]):
          if phi_2[i]==0:
              phi_2[i] = 0.000000001
          if np.abs(phi_2[i])>PI:
              phi_2[i] = PI
      eta_2 = eta_2.astype(variable_type)
      phi_2 = phi_2.astype(variable_type)
      alpha = np.arctan2(phi_2,eta_2)
      # alpha = np.arctan2(eta_2,phi_2)
      # alphas1.append(phi_2/eta_2)
      # alphas.append(alpha)
      for i in range(eta.shape[0]):
          alpha_ = - alpha[i]
          pf_vars[i, cand_eta_index, :] = eta[i,:] * np.cos(alpha_) - phi[i,:] * np.sin(alpha_)
          pf_vars[i, cand_phi_index, :] = eta[i,:] * np.sin(alpha_) + phi[i,:] * np.cos(alpha_)


def jet_image_matrix(pf, eta_jet, phi_jet, radius):
    n_eta = int((2*radius)/step_eta)
    n_phi = int((2*radius)/step_phi)
    #created the minimum edges on the 2 axes
    matrix = np.zeros((n_images,n_eta,n_phi))
    pt_pf = pf[cand_pt_index,:]
    eta_pf = pf[cand_eta_index,:]
    phi_pf = pf[cand_phi_index,:]
    pdgId_pf = np.absolute(pf[cand_pdgID_index,:]).astype(np.int)
    mask = (np.absolute(eta_pf)<radius)*(np.absolute(phi_pf)<radius)
    pt_pf = pt_pf[mask]
    eta_pf = eta_pf[mask]
    phi_pf = phi_pf[mask]
    pdgId_pf = pdgId_pf[mask]
    for i in range(len(pt_pf)):
        x = int((eta_pf[i]+radius)//step_eta)
        y = int((phi_pf[i]+radius)//step_phi)
        if pdgId_pf[i] == 130: #neutral hadron
            matrix[0,x,y] += 1
        elif pdgId_pf[i] == 211: #charged hadron
            matrix[1,x,y] += 1
            matrix[2,x,y] += pt_pf[i]/np.cosh(eta_pf[i])
            # matrix[2,x,y] += pt_pf[i]
    return matrix


@timeit
def create_image(jet_vars, pf_vars, radius):
    first = True
    first1 = True
    suncounter = 0
    for i in range(pf_vars.shape[0]):
        eta_jet = jet_vars[i,jet_eta_index]
        phi_jet = jet_vars[i,jet_phi_index]
        jet_image = jet_image_matrix(pf_vars[i,:,:], eta_jet, phi_jet, radius)
        jet_image = np.expand_dims(jet_image, axis=0)
        if first:
            jet_images_temp = jet_image
            first = False
        else:
            jet_images_temp = np.concatenate((jet_images_temp,jet_image))
            suncounter += 1
            if suncounter >10:
                suncounter = 0
                first = True
                if first1:
                    jet_images = jet_images_temp
                    first1 = False
                else:
                    jet_images = np.concatenate((jet_images,jet_images_temp))
    if suncounter !=0:
        jet_images = np.concatenate((jet_images,jet_images_temp))
    # numpy.ndarray (n_events, n_images, 2*radius, 2*radius )
    return jet_images


def preprocessing_pt_order(jet_vars, pf_vars, subjet_vars, pt_min, pt_max, isGen):
    if isGen:
        i = genjet_pt_index
    else:
        i = jet_pt_index
    mask = (jet_vars[:,i]>pt_min)*(jet_vars[:,i]<pt_max)
    jet_info_ = jet_vars[mask]
    pf_info_  = pf_vars[mask]
    subjet_info_  = subjet_vars[mask]
    # selects events in n_events according to pt cuts: Same selection applied to jet and pf-cands
    return jet_info_, pf_info_, subjet_info_


def jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    temp_time=time.time()
    images = []
    radius_= 0.8
    if "15" in radius:
        radius_ = 1.5
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name   = file_path+name_folder+bkg+"_"+radius+"/JetInfo_"+bkg+"_"+str(i)+".npy"
        file_name_1 = file_path+name_folder+bkg+"_"+radius+"/CandInfo_"+bkg+"_"+str(i)+".npy"
        file_name_2 = file_path+name_folder+bkg+"_"+radius+"/SubJetInfo_"+bkg+"_"+str(i)+".npy"
        print file_name
        if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
            continue
        isGen = 0
        if name_variable=='gen_':
            isGen = 1
        jet_vars = np.load(file_name)
        # numpy.ndarray (n_events,branches.size())
        pf_vars = np.load(file_name_1)
        subjet_vars = np.load(file_name_2)
        # numpy.ndarray (n_events,branches.size(), ncand)
        jet_vars, pf_vars, subjet_vars = preprocessing_pt_order(jet_vars, pf_vars, subjet_vars, pt_min, pt_max, isGen)
        jet_vars_ = copy.deepcopy(jet_vars)
        pf_vars_ = copy.deepcopy(pf_vars)
        subjet_vars_ = copy.deepcopy(subjet_vars)
        jet_preprocessing(pf_vars, jet_vars,subjet_vars)
        if len(jet_vars) == 0 or len(pf_vars) == 0:
            print i
            continue
        #same shape, with less n_events
        image = create_image(jet_vars, pf_vars, radius_)
        images.append(image)
    print("progress: 100%")
    print ("time needed: "+str((time.time()-temp_time))+" s")
    return images, jet_vars_, pf_vars_, subjet_vars_, jet_vars, pf_vars, subjet_vars


##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    bkg = sys.argv[3]
    radius = sys.argv[4]
    name_variable = sys.argv[5]
    pt_min = int(sys.argv[6])
    pt_max = int(sys.argv[7])
    flag_merge =int(sys.argv[8])
except Exception as e:
    file_min = 0
    file_max = 10
    bkg = "Higgs"
    radius = "AK8"
    info = ["JetInfo","CandInfo"]
    pt_min = 300
    pt_max = 500
    flag_merge = 0


file_min = 0
file_max = 5
bkg = "QCD"
radius = "AK15"
name_variable = "norm"
pt_min = 300
pt_max = 500
flag_merge = 0


translation = 1
rotation = 1

if name_variable=='norm':
    name_variable=""

# print file_min, file_max, pt_min, pt_max, name_variable, bkg, radius

file_path = out_path
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = "input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/"


alphas = []
alphas1 = []
images, jet_vars_, pf_vars_, subjet_vars_, jet_vars, pf_vars, subjet_vars = jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)




firstAdd = True

for i in range(len(images)):
  for j in range(images[i].shape[0]):
    if firstAdd:
      result = images[i][j,:,:,:]
      firstAdd = False
    else:
      result = np.add(result,images[i][j,:,:,:])

canvases, histos = plotJetImages(result,bkg+radius)



pt_min = 300
pt_max = 10000

for bkg in bkgs:
    for radius in radii:
        name_folder_output = "input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/"
        alphas = []
        alphas1 = []
        images, jet_vars_, pf_vars_, subjet_vars_, jet_vars, pf_vars, subjet_vars = jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
        images = [images[0][:1,:,:]]
        firstAdd = True
        for i in range(len(images)):
          for j in range(images[i].shape[0]):
            if firstAdd:
              result = images[i][j,:,:,:]
              firstAdd = False
            else:
              result = np.add(result,images[i][j,:,:,:])
        canvases, histos = plotJetImages(result,bkg+radius)
        for i in range(len(canvases)):
            canvases[i].Print(file_path+"plot/JetImages/"+bkg+"_"+radius+"_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_10.pdf")






jet_vars = copy.deepcopy(jet_vars_)
pf_vars = copy.deepcopy(pf_vars_)
subjet_vars = copy.deepcopy(subjet_vars_)

jet_vars.shape
pf_vars.shape
subjet_vars.shape


eta = pf_vars[:, cand_eta_index, :]
phi = pf_vars[:, cand_phi_index, :]
eta_subjet0 = subjet_vars[:, [subjet0_eta_index]]
phi_subjet0 = subjet_vars[:, [subjet0_phi_index]]
if translation:
  # eta -= eta[:,[0]]
  # phi -= phi[:,[0]]
  eta -= eta_subjet0
  phi -= phi_subjet0

if rotation:
  eta_2 = eta[:,1].astype(variable_type)
  phi_2 = phi[:,1].astype(variable_type)
  eta_2 = subjet_vars[:, subjet1_eta_index].astype(variable_type)
  phi_2 = subjet_vars[:, subjet1_phi_index].astype(variable_type)
  alpha = copy.deepcopy(np.arctan2(phi_2,eta_2))
  alpha[alpha<0] += 2*PI
  c = np.cos(alpha)
  s = np.sin(alpha)
  angles = np.array((eta,phi)).swapaxes(0,1).swapaxes(1,2)
  R = np.array(((c,-s), (s, c))).swapaxes(0,1).swapaxes(0,2)
  angles = np.matmul(angles, R)
  pf_vars[:, cand_eta_index, :] = angles[:,:,0]
  pf_vars[:, cand_phi_index, :] = angles[:,:,1]
