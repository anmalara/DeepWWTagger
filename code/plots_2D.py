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
  for n in range(n_images):
    c = tdrCanvas(process+"_"+images_dict[n], -radius_, radius_, -radius_, radius_, "#eta", "#phi", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
    gStyle.SetOptStat(0)
    c.SetRightMargin(1)
    h = TH2F( process+"_"+images_dict[n], process+"_"+images_dict[n], Nbins, -radius_, radius_, Nbins, -radius_, radius_)
    for i in range(1,Nbins):
      for j in range(1,Nbins):
        h.SetBinContent(i,j,array[n,i,j])
    h.Draw("colz")
    time.sleep(2)
    c.Print(output_path+process+"_"+images_dict[n]+".pdf")
    c.Print(output_path+process+"_"+images_dict[n]+".png")
    c.Print(output_path+process+"_"+images_dict[n]+".root")


def addEvents(file_name):
  file_ = np.load(file_name)
  firstAdd = True
  for i in range(file_.shape[0]):
    if firstAdd:
      result = file_[i,:,:,:]
      firstAdd = False
    else:
      result = np.add(result,file_[i,:,:,:])
  return result

@timeit
def addFiles(min, max, bkg, radius, pt):
  firstEvent = True
  for i in range(min, max):
    file_name = out_path+"input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/"+"JetImage_matrix_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+pt+".npy"
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
