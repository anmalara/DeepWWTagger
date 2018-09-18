import sys
import os
import os.path
import glob
import time
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



def matrix_norm(matrix):
  sum = 0
  for x in range(matrix.shape[0]):
    for y in range(matrix.shape[1]):
      sum = matrix[x,y]*matrix[x,y]
  matrix = matrix / sum


def jet_preprocessing(cand_info, jet_info):
  if translation:
    print "translation"
    cand_info[:, branch_names_candidate.index("CandEta"), :] -= cand_info[:, branch_names_candidate.index("CandEta"), [0]]
    cand_info[:, branch_names_candidate.index("CandPhi"), :] -= cand_info[:, branch_names_candidate.index("CandPhi"), [0]]
  if rotation:
    print "rotation"
    eta_2 = cand_info[:, branch_names_candidate.index("CandEta"), 1]
    phi_2 = cand_info[:, branch_names_candidate.index("CandPhi"), 1]
    alpha = np.arctan(phi_2/eta_2)
    eta = cand_info[:, branch_names_candidate.index("CandEta"), :]
    phi = cand_info[:, branch_names_candidate.index("CandPhi"), :]

    eta = eta * np.cos(alpha) + phi * np.sin(alpha)
    phi = eta * np.sin(alpha) - phi * np.cos(alpha)
    # pt_1 = jet_info[:, branch_names_jet.index("ptSub1")]
    # eta_1 = jet_info[:, branch_names_jet.index("etaSub1")]
    # phi_1 = jet_info[:, branch_names_jet.index("phiSub1")]
    # px_1 = pt_1* np.cos(phi_1)
    # py_1 = pt_1* np.sin(phi_1)
    # pz_1 = pt_1* np.sinh(eta_1)
    # theta_1 = np.arctan(py_1/pz_1) + math.pi/2.
    # py_n = cand_info[:, branch_names_candidate.index("candPy"), :]
    # pz_n = cand_info[:, branch_names_candidate.index("candPz"), :]
    # cand_info[:, branch_names_candidate.index("candPy"), :] = py_n*np.cos(theta_1) - pz_n*np.sin(theta_1)
    # cand_info[:, branch_names_candidate.index("candPz"), :] = py_n*np.sin(theta_1) - pz_n*np.cos(theta_1)


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
    pt_pf = pf[branch_names_candidate.index("CandPt"),i]
    eta_pf = pf[branch_names_candidate.index("CandEta"),i] - eta_jet
    phi_pf = pf[branch_names_candidate.index("CandPhi"),i] - phi_jet
    pdgId_pf = int(abs(pf[branch_names_candidate.index("CandPdgId"),i]))
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
    if pdgId_pf == 130:
      #neutral hadron
      matrix[0,x,y] += 1
    elif pdgId_pf == 211:
      #charged hadron
      matrix[1,x,y] += 1
      matrix[2,x,y] += pt_pf
  return matrix


def create_image(jet_info, pf_info, radius):
  for i in range(pf_info.shape[0]):
    eta_jet = jet_info[i,branch_names_jet.index("jetEta")]
    phi_jet = jet_info[i,branch_names_jet.index("jetPhi")]
    jet_image = jet_image_matrix(pf_info[i,:,:], eta_jet, phi_jet, radius)
    jet_image = jet_image.reshape((1,jet_image.shape[0],jet_image.shape[1],jet_image.shape[2]))
    if i==0:
      jet_images = jet_image
    else:
      jet_images = np.concatenate((jet_images,jet_image))
  # numpy.ndarray (n_events, n_images, 2*radius, 2*radius )
  return jet_images


def preprocessing_pt_order(jet_info, pf_info, pt_min, pt_max, isGen):
  if isGen:
    i = branch_names_gen_jet.index("GenJetPt")
  else:
    i = branch_names_jet.index("jetPt")
  jet_info_ = jet_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
  pf_info_ = pf_info[(jet_info[:,i]>pt_min)*(jet_info[:,i]<pt_max)]
  # selects events in n_events according to pt cuts: Same selection applied to jet and pf-cands
  return jet_info_, pf_info_


def jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
  temp_time=time.time()
  radius_= 0.8
  images = []
  if "15" in radius:
    radius_ = 1.5
  for i in range(file_min, file_max):
    if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
    file_name   = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
    file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"cand_var_"+bkg+"_"+str(i)+".npy"
    if (not os.path.isfile(file_name) or not os.path.isfile(file_name_1)):
      continue
    isGen = 0
    if name_variable=='gen_':
      isGen = 1
    jet_vars = np.load(file_name)
    # numpy.ndarray (n_events,branches.size())
    pf_vars = np.load(file_name_1)
    # numpy.ndarray (n_events,branches.size(), ncand)
    jet_vars, pf_vars = preprocessing_pt_order(jet_vars, pf_vars, pt_min, pt_max, isGen)
    jet_preprocessing(pf_vars, jet_vars)
    if len(jet_vars) == 0 or len(pf_vars) == 0:
      print i
      continue
    #same shape, with less n_events
    image = create_image(jet_vars, pf_vars, radius_)
    images.append(image)
  print("progress: 100%")
  print ("time needed: "+str((time.time()-temp_time))+" s")
  return images


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
    name_variable = "norm"
    pt_min = 300
    pt_max = 500
    flag_merge = 0


file_min = 0
file_max = 100
bkg = "Higgs"
radius = "AK15"
name_variable = "norm"
pt_min = 300
pt_max = 500
flag_merge = 0


translation = 1
rotation = 0

if name_variable=='norm':
    name_variable=""

# print file_min, file_max, pt_min, pt_max, name_variable, bkg, radius

file_path = out_path
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = "input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/"



images = jetImage_inputFiles(file_path, name_folder, name_variable, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)

firstAdd = True

for i in range(len(images)):
  for j in range(images[i].shape[0]):
    if firstAdd:
      result = images[i][j,:,:,:]
      firstAdd = False
    else:
      result = np.add(result,images[i][j,:,:,:])

canvases, histos = plotJetImages(result,bkg+radius)



import copy
i = 10

file_name   = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"jet_var_"+bkg+"_"+str(i)+".npy"
file_name_1 = file_path+name_folder+bkg+"_"+radius+"/"+name_variable+"cand_var_"+bkg+"_"+str(i)+".npy"
isGen = 0
jet_vars = np.load(file_name)
pf_vars = np.load(file_name_1)
jet_vars, pf_vars = preprocessing_pt_order(jet_vars, pf_vars, pt_min, pt_max, isGen)

cand_info = copy.deepcopy(pf_vars)

eta = cand_info[:, branch_names_candidate.index("CandEta"), :]
phi = cand_info[:, branch_names_candidate.index("CandPhi"), :]

eta -= cand_info[:, branch_names_candidate.index("CandEta"), [0]]
phi -= cand_info[:, branch_names_candidate.index("CandPhi"), [0]]


eta_2 = cand_info[:, branch_names_candidate.index("CandEta"), [1]]
phi_2 = cand_info[:, branch_names_candidate.index("CandPhi"), [1]]
alpha = np.arctan(phi_2/eta_2)


eta = eta * np.cos(alpha) + phi * np.sin(alpha)
phi = eta * np.sin(alpha) - phi * np.cos(alpha)
