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
from math import *
from math import pi as PI

from variables import *
from Prepocessing import *

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from tdrstyle_all import *

colors = [kBlack, kRed+1, kBlue-4, kGreen-2, kOrange, kMagenta, kViolet-3, kCyan, kSpring, kTeal, kYellow+1, kPink+10, kAzure+7, kAzure+1, kRed+3, kGray]

@timeit
def plotJetImages(array, process, output_path="./",):
    if "15" in process:
        Radius= 1.5
    elif "8" in process:
        Radius= 0.8
    print process, Radius
    Nbins = int(2*Radius/step_eta)
    canvases = []
    histos = []
    for n in range(n_images):
        c = tdrCanvas(process+"_"+images_dict[n], -Radius, Radius, -Radius, Radius, "#eta", "#phi", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
        gStyle.SetOptStat(0)
        c.SetRightMargin(1)
        c.SetLogz(1)
        h = TH2F( process+"_"+images_dict[n], process+"_"+images_dict[n], Nbins, -Radius, Radius, Nbins, -Radius, Radius)
        for i in range(1,Nbins):
            for j in range(1,Nbins):
                h.SetBinContent(i,j,array[n,i,j])
        h.Draw("colz")
        histos.append(h)
        canvases.append(c)
    return canvases, histos


def jet_preprocessing(cand_vars):
    translation = 1
    rotation = 1
    eta = cand_vars[:, CandEta_index, :]
    phi = cand_vars[:, CandPhi_index, :]
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
        cand_vars[:, CandEta_index, :] = angles[:,:,0]
        cand_vars[:, CandPhi_index, :] = angles[:,:,1]


def jet_image_matrix(pf, eta_jet, phi_jet, Radius):
    n_eta = int((2*Radius)/step_eta)
    n_phi = int((2*Radius)/step_phi)
    #created the minimum edges on the 2 axes
    matrix = np.zeros((n_images,n_eta,n_phi))
    pt_pf = pf[CandPt_index,:]
    eta_pf = pf[CandEta_index,:]
    phi_pf = pf[CandPhi_index,:]
    pdgId_pf = np.absolute(pf[CandPdgId_index,:]).astype(np.int)
    mask = (np.absolute(eta_pf)<Radius)*(np.absolute(phi_pf)<Radius)
    pt_pf = pt_pf[mask]
    eta_pf = eta_pf[mask]
    phi_pf = phi_pf[mask]
    pdgId_pf = pdgId_pf[mask]
    for i in range(len(pt_pf)):
        x = int((eta_pf[i]+Radius)//step_eta)
        y = int((phi_pf[i]+Radius)//step_phi)
        if pdgId_pf[i] == 130: #neutral hadron
            matrix[0,x,y] += 1
        elif pdgId_pf[i] == 211: #charged hadron
            matrix[1,x,y] += 1
            matrix[2,x,y] += pt_pf[i]/np.cosh(eta_pf[i])
            # matrix[2,x,y] += pt_pf[i]
    return matrix

def create_image(jet_vars, cand_vars, Radius):
    jet_images = []
    suncounter = 0
    for i in range(cand_vars.shape[0]):
        eta_jet = jet_vars[i,jetEta_index]
        phi_jet = jet_vars[i,jetPhi_index]
        jet_image = jet_image_matrix(cand_vars[i,:,:], eta_jet, phi_jet, Radius)
        jet_image = np.expand_dims(jet_image, axis=0)
        jet_images.append(jet_image)
    if len(jet_images)>0:
        jet_images = np.concatenate(jet_images)
    else:
        jet_images = np.empty((n_images,int((2*Radius)/step_eta),int((2*Radius)/step_phi)))
    # numpy.ndarray (n_events, n_images, 2*Radius, 2*Radius )
    return jet_images

@timeit
def jetImage_inputFiles(folder_input, folder_output, file_min, file_max, bkg, radius, pt_min, pt_max):
    infos = ["JetInfo","CandInfo","SubJetInfo"]
    Radius= 0.8
    Found = True
    if "15" in radius:
        Radius = 1.5
    for i in range(file_min, file_max):
        # if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        Vars = {}
        for info in infos:
            file_name = folder_input+bkg+"_"+radius+"/"+info+"/"+info+"_"+str(i)+".npy"
            if not os.path.isfile(file_name):
                with open(folder_output+"problem_"+bkg+"_"+radius+".txt","a") as F:
                    F.write("Missing "+bkg+"_"+radius+"_"+info+"_"+str(pt_min)+"_"+str(pt_max)+" "+str(i)+"\n")
                Found = False
                break
            Vars[info] = np.load(file_name)
        if not Found:
            continue
        # selects events in n_events according to pt cuts: Same selection applied to jet and pf-cands
        preprocessing_pt_selection(Vars, "JetInfo", pt_min, pt_max)
        jet_preprocessing(Vars["CandInfo"])
        image = create_image(Vars["JetInfo"], Vars["CandInfo"], Radius)
        image = image.astype(variable_type)
        name = folder_output+"JetImage_"+infos[0]+"_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        print image.shape, name
        np.save(name, image)


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
    pt_min = int(sys.argv[5])
    pt_max = int(sys.argv[6])
except Exception as e:
    file_min = 10
    file_max = 20
    bkg = "Higgs"
    radius = "AK8"
    pt_min = 300
    pt_max = 500

folder_input  = out_path+"input_varariables/NTuples_Tagger/"
folder_output = out_path+"input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/"

jetImage_inputFiles(folder_input, folder_output, file_min, file_max, bkg, radius, pt_min, pt_max)

# images = jetImage_inputFiles(folder_input, folder_output, file_min, file_max, infos, bkg, radius, pt_min, pt_max)
# firstAdd = True
#
# for i in range(len(images)):
#   for j in range(images[i].shape[0]):
#     if firstAdd:
#       result = images[i][j,:,:,:].astype(np.float64)
#       firstAdd = False
#     else:
#       result = np.add(result,images[i][j,:,:,:])
#
# canvases, histos = plotJetImages(result,bkg+radius)
