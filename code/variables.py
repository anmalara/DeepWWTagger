import os
import os.path
import sys
import glob
import numpy as np

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")

# from Utils import *

ncand = 40
n_images = 3
OutOfRangeValue = np.int_(-1000000)
EmptyValue = np.int_(0)


images_dict = {0:"neutral_hadron", 1:"charged_hadron", 2:"pt_pf"}
step_eta = 0.1
step_phi = 0.1
selection_cuts = "((abs(jetEta)<=2.7)&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1))||(abs(jetEta)<2.4&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1)&&(jetChf>0)&&(jetChm>0)&&(jetElf<0.8))"
#
# # those are numbers
# branch_name_EventInfo       = ["runNo", "evtNo", "lumi", "nvtx", "nJets", "nGenJets", "nBJets", "pvRho", "pvz", "pvchi2", "pvndof", "rho", "ht", "met", "metGen", "metSig", "metGenSig", "npu", "genEvtWeight", "lheOriginalXWGTUP" ]
# # those are vector
# branch_names_JetInfo        = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
# branch_names_GenJetInfo     = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]
# branch_name_JetExtraInfo    = ["jetChf", "jetNhf", "jetMuf", "jetPhf", "jetElf", "jetIsBtag", "jetFlavor", "jetFlavorHadron", "MatchedLeptons", "MatchedHiggs", "WMinusLep", "WPlusLep", "jetCorr", "jetUnc", "ncandidates",]
# branch_name_GenJetExtraInfo = ["GenmassSub0", "GenmassSub1", "GenptSub0", "GenptSub1", "GenetaSub0", "GenetaSub1", "GenphiSub0", "GenphiSub1", "nGencandidates"]
# branch_name_SubJetInfo      = ["btagSub0", "btagSub1", "massSub0", "massSub1", "ptSub0", "ptSub1", "etaSub0", "etaSub1", "phiSub0", "phiSub1", "flavorSub0", "flavorSub1", "flavorHadronSub0", "flavorHadronSub1", "nSubJets", "nBSubJets"]
# # those are vector of vector
# branch_names_CandInfo       = ["CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"]
# branch_names_GenCandInfo    = ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"]
# branch_name_CandExtraInfo   = ["CandPvAssociationQuality", "CandDZAssociatedPV", "CandSubJetPart"]
#
# # seemed not to be used
# branch_name_EventExtraInfo  = ["triggerBit", "triggerPre", "scaleWeights", "pdfWeights"]

branch_names_dict = { "EventInfo"       : ["runNo", "evtNo", "lumi", "nvtx", "nJets", "nGenJets", "nBJets", "pvRho", "pvz", "pvchi2", "pvndof", "rho", "ht", "met", "metGen", "metSig", "metGenSig", "npu", "genEvtWeight", "lheOriginalXWGTUP" ],
                     # those are vector
                     "JetInfo"          : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"],
                     "GenJetInfo"       : ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"],
                     "JetExtraInfo"     : ["jetChf", "jetNhf", "jetMuf", "jetPhf", "jetElf", "jetIsBtag", "jetFlavor", "jetFlavorHadron", "MatchedLeptons", "MatchedHiggs", "WMinusLep", "WPlusLep", "jetCorr", "jetUnc", "ncandidates",],
                     "SubGenJetInfo"    : ["GenmassSub0", "GenmassSub1", "GenptSub0", "GenptSub1", "GenetaSub0", "GenetaSub1", "GenphiSub0", "GenphiSub1", "nGencandidates"],
                     "SubJetInfo"       : ["btagSub0", "btagSub1", "massSub0", "massSub1", "ptSub0", "ptSub1", "etaSub0", "etaSub1", "phiSub0", "phiSub1", "flavorSub0", "flavorSub1", "flavorHadronSub0", "flavorHadronSub1", "nSubJets", "nBSubJets"],
                     # those are vector of vector
                     "CandInfo"         : ["CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"],
                     "GenCandInfo"      : ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"],
                     "CandExtraInfo"    : ["CandPvAssociationQuality", "CandDZAssociatedPV", "CandSubJetPart"],
                     # seemed not to be used
                     "EventExtraInfo"   : ["triggerBit", "triggerPre", "scaleWeights", "pdfWeights"]}

out_path  = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"

bkgs = ["Higgs", "QCD", "Top"]
radii = ["AK8", "AK15", "CA15"]

original_paths_dict = {"Higgs"  : "/pnfs/desy.de/cms/tier2/store/user/pgunnell/HiggsZProduction/MC-CandidatesP8-Higgs-GenInfos-v4/180413_072857/",
                       "QCD"    : "/pnfs/desy.de/cms/tier2/store/user/pgunnell/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/MC-CandidatesP8-QCD-GenInfos-v4/180413_073311/",
                       "Top"    : "/pnfs/desy.de/cms/tier2/store/user/pgunnell/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MC-CandidatesP8-Top-GenInfos-v5/180503_120147/"}

# @timeit
def countOriginalFiles():
  missing_items = []
  maxima = []
  global bkgs
  for bkg in bkgs:
    path = original_paths_dict[bkg]
    maximum = 0
    missing = []
    for i in range(0,10000):
      subCounter = "000"+str(i/1000)+"/"
      path_ = path + subCounter
      name_variable = "flatTreeFile"+bkg+"-nano_"
      file_name = path+subCounter+name_variable+str(i)+".root"
      if os.path.isfile(file_name):
        maximum = i
      else:
        missing.append(str(i))
    for el in missing[:]:
      # print el, int(el)
      if int(el) > maximum:
        missing.remove(el)
    maxima.append(maximum)
    missing_items.append(missing)
  return dict(zip(bkgs,zip(maxima, missing_items)))


files_dictionary = countOriginalFiles()
