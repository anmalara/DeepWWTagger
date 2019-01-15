import sys
import os
import os.path
import glob
import time
from copy import deepcopy
import numpy as np
from ROOT import TFile, TCanvas, TLegend, TH1F, TH2F, TColor, TAxis
from ROOT import kWhite, kBlack, kGray, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan, kOrange, kSpring, kTeal, kAzure, kViolet, kPink
from ROOT import kNone, gStyle
from root_numpy import *
from math import pi as PI

from variables import *

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from tdrstyle_all import *

colors = [kBlack, kRed+1, kBlue-4, kGreen-2, kOrange, kMagenta, kViolet-3, kCyan, kSpring, kTeal, kYellow+1, kPink+10, kAzure+7, kAzure+1, kRed+3, kGray]

@timeit
def plotJetImages(array, process, output_path="./",):
  if "15" in process:
    Radius= 1.5
  elif "8" in process:
    Radius= 0.8
  Nbins = int(2*Radius/step_eta)
  print process, Radius, Nbins
  canvases = []
  histos = []
  for n in range(n_images):
    c = tdrCanvas(process+"_"+images_dict[n], -Radius, Radius, -Radius, Radius, "#eta", "#phi", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
    gStyle.SetOptStat(0)
    c.SetRightMargin(1)
    c.SetLogz(1)
    h = TH2F( process+"_"+images_dict[n], process+"_"+images_dict[n], Nbins, -Radius, Radius, Nbins, -Radius, Radius)
    for i in range(0,Nbins):
      for j in range(0,Nbins):
        h.SetBinContent(i+1,j+1,array[n,i,j])
    h.Draw("colz")
    histos.append(h)
    canvases.append(c)
    return canvases, histos
    # c.Print(output_path+process+"_"+images_dict[n]+".pdf")
    # c.Print(output_path+process+"_"+images_dict[n]+".png")
    # c.Print(output_path+process+"_"+images_dict[n]+".root")


def addEvents(images):
  firstAdd = True
  for i in range(len(images)):
      for j in range(images[i].shape[0]):
          if firstAdd:
              firstAdd = False
              overlap_images = deepcopy(images[i][j,:,:,:])
          else:
              overlap_images = np.add(overlap_images,images[i][j,:,:,:],dtype=np.float64)
  return overlap_images

@timeit
def addFiles(min, max, bkg, radius, pt):
  firstEvent = True
  for i in range(min, max):
    file_name = out_path+"input_varariables/NTuples_Tagger/Inputs/"+bkg+"_"+radius+"/JetImage/file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
    if not os.path.isfile(file_name):
      continue
    temp = addEvents(file_name)
    if firstEvent:
      array = temp
      firstEvent = False
    else:
      array = np.add(array,temp)
  return array



########################
#                      #
#     Main Program     #
#                      #
########################

output_path = out_path+"plot/JetImages/Initial/"
if not os.path.isdir(output_path):
  os.makedirs(output_path)
for pt in ["300_500", "500_10000"]:
  for bkg in bkgs:
    for radius in radii:
      process = bkg+radius+"_pt_"+pt
      print process
      # for i in range(files_dictionary[bkg][0]):
      array = addFiles(0, 100, bkg, radius, pt)
      # array = addFiles(0, files_dictionary[bkg][0], bkg, radius, pt)
      plotJetImages(array, process, output_path)
      del array
