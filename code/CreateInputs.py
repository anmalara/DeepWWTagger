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
from Prepocessing import *
from ImageCreation import *
from AdditionalVariablesCreation import *

def LoadVariables(folder_input,bkg,radius,file_index):
    Vars = {}
    for info in branch_names_dict.keys():
        file_name = folder_input+bkg+"_"+radius+"/"+info+"/"+info+"_"+str(file_index)+".npy"
        try:
            Vars[info] = np.load(file_name)
        except:
            pass
    return Vars

@timeit
def SaveVariables(folder_output, bkg, radius, pt_min, pt_max, file_index, Vars, JetImage, JetVariables):
    file_name = "/file_"+str(file_index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
    for info in branch_names_dict.keys():
        np.save(folder_output+bkg+"_"+radius+"/"+info+file_name, Vars[info])
    np.save(folder_output+bkg+"_"+radius+"/JetImage/"+file_name, JetImage)
    np.save(folder_output+bkg+"_"+radius+"/JetVariables/"+file_name, JetVariables)


@timeit
def CreateInputs(folder_input, file_min, file_max, bkg, radius, pt_min, pt_max):
    Radius= 0.8
    if "15" in radius:
        Radius = 1.5
    for file_index in range(file_min, file_max):
        Vars = LoadVariables(folder_input,bkg,radius,file_index)
        if len(Vars.keys())!= len(branch_names_dict.keys()):
            continue
        Preprocessing(Vars,pt_min, pt_max)
        JetImage = CreateImage(Vars, Radius)
        JetVariables = CreateJetVariables(Vars)
        SaveVariables(folder_output, bkg, radius, pt_min, pt_max, file_index, Vars, JetImage, JetVariables)




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
folder_output = out_path+"input_varariables/NTuples_Tagger/Inputs/"

CreateInputs(folder_input, file_min, file_max, bkg, radius, pt_min, pt_max)
