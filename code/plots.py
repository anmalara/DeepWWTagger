import sys
import os
import os.path
import glob
import time
import numpy as np
from ROOT import TFile, TCanvas, TLegend, TH1F, TH2F, TColor, TAxis
from ROOT import kWhite, kBlack, kGray, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan, kOrange, kSpring, kTeal, kAzure, kViolet, kPink
from ROOT import kNone
from root_numpy import *
from math import pi as PI

from variables import *

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
from tdrstyle_all import *

colors = [kBlack, kRed+1, kBlue-4, kGreen-2, kOrange, kMagenta, kViolet-3, kCyan, kSpring, kTeal, kYellow+1, kPink+10, kAzure+7, kAzure+1, kRed+3, kGray]

def get_binInfo(branch="Pt"):
  if "jetPt".lower() in branch.lower():
    return 300, 0, 3000, 0.00001, 1., 0, 1
  elif "jetEta".lower() in branch.lower():
    return 100, -PI, PI, 0.00001, 0.05, 0, 0
  elif "jetPhi".lower() in branch.lower():
    return 100, -PI, PI, 0.00001, 0.03, 0, 0
  elif "jetEnergy".lower() in branch.lower():
    return 500, 0, 5000, 0.00001, 0.35, 0, 1
  elif "jetBtag".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.07, 0, 0
  elif "SoftDrop".lower() in branch.lower():
    return 100, 0, 500, 0.00001, 0.3, 0, 0
  elif "jetMass".lower() in branch.lower():
    return 100, 0, 500, 0.00001, 0.14, 0, 0
  elif "jetTau1".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.1, 0, 0
  elif "jetTau2".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.15, 0, 0
  elif "jetTau3".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.2, 0, 0
  elif "jetTau4".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.25, 0, 0
  elif "isB".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 0.25, 0, 0
  elif "CandEnergy".lower() in branch.lower():
    return 150, 0, 1500, 0.00001, 1, 0, 1
  elif "CandPx".lower() in branch.lower():
    return 100, -1500, 1500, 0.00001, 1.0, 0, 1
  elif "CandPy".lower() in branch.lower():
    return 100, -1500, 1500, 0.00001, 1.0, 0, 1
  elif "CandPz".lower() in branch.lower():
    return 100, -2000, 2000, 0.00001, 1.0, 0, 1
  elif "CandPt".lower() in branch.lower():
    return 150, 0, 1500, 0.00001, 1., 0, 1
  elif "CandEta".lower() in branch.lower():
    return 100, -2*PI, 2*PI, 0., 0.1, 0, 0
  elif "CandPhi".lower() in branch.lower():
    return 50, -PI, PI, 0., 0.06, 0, 0
  elif "CandPdgId".lower() in branch.lower():
    return 500, -250, 250, 0.01, 1, 0, 1
  elif "CandMass".lower() in branch.lower():
    return 200, -1, 1, 0.00001, 1., 0, 1
  elif "CandDXY".lower() in branch.lower():
    return 100, -20, 20, 0.00001, 10, 0, 1
  elif "CandDZ".lower() in branch.lower():
    return 100, -0.4, 0.4, 0.000001, 10, 0, 1
  elif "CandPuppiWeight".lower() in branch.lower():
    return 100, 0, 1, 0.00001, 50, 0, 1
  else:
    return 100, 0, 1000, 0.00001, 1., 0, 0
    # return N_bins, bin_min, bin_max, min, max, isLogx, isLogy



@timeit
def plotJetVariables(arrays=[], array_names=["Higgs"], output_path="./", branch_names=["jetPt", "jetEta"], isCand=False):
  for index, branch in enumerate(branch_names):
    print branch
    N_bins, bin_min, bin_max, max_, min_, isLogx, isLogy = get_binInfo(branch)
    c = tdrCanvas(branch, bin_min, bin_max, max_, min_, branch, "A.U.", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
    c.SetLogx(isLogx)
    c.SetLogy(isLogy)
    leg = tdrLeg(0.55, 0.5, 0.9, 0.9, textSize=0.025)
    tdrHeader(leg, branch)
    histos = []
    for index_array, array in enumerate(arrays):
      h = TH1F( branch+array_names[index_array], branch+array_names[index_array], N_bins, bin_min, bin_max)
      if isCand:
        for i in range(array.shape[2]):
          fill_hist(h, array[:,index,i])
      else:
        fill_hist(h, array[:,index])
      h.SetLineWidth(3)
      if h.Integral()>0:
        h.Scale(1./h.Integral())
      tdrDraw(h, "hist", mcolor=colors[index_array+1], lcolor=colors[index_array+1], fstyle=0, fcolor=colors[index_array+1])
      leg.AddEntry(h, array_names[index_array] + ", Entries: "+str(round(float(h.GetEntries())/1000000,3))+" M","l")
      histos.append(h)
    c.Print(output_path+branch+".pdf")
    c.Print(output_path+branch+".png")
    c.Print(output_path+branch+".root")


@timeit
def runOverInputs(arrays,array_names, branch_names, isCand):
  output_path = out_path+common_path+"all/"
  if not os.path.isdir(output_path):
    os.makedirs(output_path)
  plotJetVariables(arrays, array_names, output_path, branch_names, isCand)
  for bkg in bkgs:
    temp_array_names = [array_names[index] for index, test in enumerate(array_names) if bkg in test]
    temp_arrays = [arrays[index] for index, test in enumerate(array_names) if bkg in test]
    print temp_array_names
    output_path = out_path+common_path+bkg+"/"
    if not os.path.isdir(output_path):
      os.makedirs(output_path)
    plotJetVariables(temp_arrays, temp_array_names, output_path, branch_names, isCand)
  for radius in radii:
    temp_array_names = [array_names[index] for index, test in enumerate(array_names) if radius in test]
    temp_arrays = [arrays[index] for index, test in enumerate(array_names) if radius in test]
    print temp_array_names
    output_path = out_path+common_path+radius+"/"
    if not os.path.isdir(output_path):
      os.makedirs(output_path)
    plotJetVariables(temp_arrays, temp_array_names, output_path, branch_names, isCand)


def resetError(arrays):
  for array in arrays:
    for x in range(array.shape[0]):
      for y in range(array.shape[1]):
        try:
          len(array[x,y])
          array[x,y] = 100000
        except:
          pass


@timeit
def addFiles(path, var, bkg):
  firstEvent = True
  for i in range(files_dictionary[bkg][0]):
    file_name = path+var+"_var_"+bkg+"_"+str(i)+".npy"
    if os.path.isfile(file_name):
      file = np.load(file_name)
      if firstEvent:
        array = file
        firstEvent = False
      else:
        array = np.concatenate((array,file))
      if len(array)>1000000:
        break
  return array


########################
#                      #
#     Main Program     #
#                      #
########################

# for var in ["jet", "cand", "gen_jet", "gen_cand"]:
for var in ["cand", "gen_jet", "gen_cand"]:
  arrays = []
  array_names = []
  for bkg in bkgs:
    for radius in radii:
      path = out_path+"input_varariables/NTuples_Tagger/"+bkg+"_"+radius+"/"
      array_name = bkg+radius
      array_names.append(array_name)
      print bkg, radius
      array = addFiles(path, var, bkg)
      print array.shape
      arrays.append(array)
      del array
  if var == "jet":
    branch_names = branch_names_jet
    # branch_names = ["jetPt"]
    isCand = False
  if var == "gen_jet":
    branch_names = branch_names_gen_jet
    # branch_names = ["GenJetPt"]
    isCand = False
  if var == "cand":
    branch_names = branch_names_candidate
    # branch_names = ["CandPt"]
    isCand = True
  if var == "gen_cand":
    branch_names = branch_names_gen_cand
    # branch_names = ["GenCandPt"]
    isCand = True
  if not isCand:
    resetError(arrays)
  common_path = "./plot/"
  print var
  print branch_names
  runOverInputs(arrays, array_names, branch_names, isCand)


for var in ["", "gen_"]:
  for pt in ["300_500", "500_10000"]:
    arrays = []
    array_names = []
    for bkg in bkgs:
      for radius in radii:
        for path in glob.glob(out_path+"input_varariables/NTuples_Tagger/Sequential/Sequential_"+var+"input_variable_"+bkg+"_"+radius+"*"+pt+"*"):
          if not os.path.isfile(path):
            continue
          array_name = bkg+radius+"_pt_"+pt
          print array_name
          try:
            array = np.load(path)
            print array.shape
            arrays.append(array)
            array_names.append(array_name)
          except:
            continue
    isCand = False
    if var == "":
      branch_names = branch_names_jet
    if var == "gen_":
      branch_names = branch_names_gen_jet
    resetError(arrays)
    common_path = "./plot/Sequential/pt_"+pt+"/"
    print var
    print branch_names
    runOverInputs(arrays, array_names, branch_names, isCand)
