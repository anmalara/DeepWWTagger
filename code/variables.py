import os
import os.path
import sys
from glob import glob
import numpy as np

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94/CMSSW_9_4_1/src/UHH2/PersonalCode/")
try:
    from Utils import *
except:
    print "No Utils imported"

def countOriginalFiles(bkgs,original_paths_dict):
    files_dictionary = {}
    for bkg in bkgs:
        sum_ = 0
        element = []
        for index, path in enumerate(original_paths_dict[bkg]):
            for el in glob(path+"000*/*.root"):
                element.append(10000*index+int(el[el.rfind("_")+1:-5]))
        files_dictionary[bkg] = {"elements" : sorted((element)), "maximum": max(sorted(element))}
    return files_dictionary

def MakeDirs(outputdir):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

def MakeOutDirs(bkgs, radii, branch_names_dict,out_path):
    for bkg in bkgs:
        for radius in radii:
            for info in branch_names_dict:
                MakeDirs(out_path+"input_varariables/NTuples_Tagger/"+bkg+"_"+radius+"/"+info)
                MakeDirs(out_path+"input_varariables/NTuples_Tagger/Inputs/"+bkg+"_"+radius+"/"+info)
                MakeDirs(out_path+"input_varariables/NTuples_Tagger/Inputs/"+bkg+"_"+radius+"/JetImage")
                MakeDirs(out_path+"input_varariables/NTuples_Tagger/Inputs/"+bkg+"_"+radius+"/JetVariables")


ncand_tosave = 200
ncand = 30
n_images = 3
OutOfRangeValue = np.int_(-1000000)
EmptyValue = np.int_(0)
variable_type = np.float16

images_dict = {0:"neutral_hadron", 1:"charged_hadron", 2:"pt_pf"}
step_eta = 0.1
step_phi = 0.1
selection_cuts = "((abs(jetEta)<=2.7)&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1))||(abs(jetEta)<2.4&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1)&&(jetChf>0)&&(jetChm>0)&&(jetElf<0.8))"

branch_names_dict = { "EventInfo"       : ["runNo", "evtNo", "lumi", "nvtx", "nJets", "nGenJets", "nBJets", "pvRho", "pvz", "pvchi2", "pvndof", "rho", "ht", "met", "metGen", "metSig", "metGenSig", "npu", "genEvtWeight", "lheOriginalXWGTUP" ],
                     # those are vector
                     "JetInfo"          : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4", "ncandidates"],
                     "GenJetInfo"       : ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4", "nGencandidates"],
                     "JetExtraInfo"     : ["jetChf", "jetNhf", "jetMuf", "jetPhf", "jetElf", "jetIsBtag", "jetFlavor", "jetFlavorHadron", "MatchedLeptons", "MatchedHiggs", "WMinusLep", "WPlusLep", "jetCorr", "jetUnc"],
                     "SubGenJetInfo"    : ["GenmassSub0", "GenmassSub1", "GenptSub0", "GenptSub1", "GenetaSub0", "GenetaSub1", "GenphiSub0", "GenphiSub1"],
                     "SubJetInfo"       : ["btagSub0", "btagSub1", "massSub0", "massSub1", "ptSub0", "ptSub1", "etaSub0", "etaSub1", "phiSub0", "phiSub1", "flavorSub0", "flavorSub1", "flavorHadronSub0", "flavorHadronSub1", "nSubJets", "nBSubJets"],
                     # those are vector of vector
                     "CandInfo"         : ["CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"],
                     "GenCandInfo"      : ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"],
                     # "CandExtraInfo"    : ["CandPvAssociationQuality", "CandDZAssociatedPV", "CandSubJetPart"],
                     # seemed not to be used
                     # "EventExtraInfo"   : ["triggerBit", "triggerPre", "scaleWeights", "pdfWeights"]
                     }

Info_dict = branch_names_dict.copy()
Info_dict["JetVariables"] = ["SumCandPx", "SumCandPy", "SumCandPz", "SumCandPt", "SumAllPt", "SumCandEnergy", "SumCandMass", "jetTau21", "jetTau31", "jetTau41", "jetTau32", "jetTau42", "jetTau43"]
Info_dict["JetImage"] = ["NeutralHadron", "ChargedHadron", "PtPf"]

for info in branch_names_dict:
    for var in branch_names_dict[info]:
        globals()[var+"_index"] = branch_names_dict[info].index(var)

out_path  = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"

bkgs = ["Higgs", "QCD", "Top", "DY", "WJets", "ZZ", "WZ"]
radii = ["AK8", "AK15", "CA15"]

pts_min = [300,500]
pts_max = [500,10000]

tree_dict = {"Higgs" : "flatTreeFileHiggs-nano_",
             "QCD"   : "flatTreeFileQCD-nano_",
             "Top"   : "flatTreeFileTop-nano_",
             "DY"    : "flatTreeFileDY-nano_",
             "WJets" : "flatTreeFileW-nano_",
             "ZZ"    : "flatTreeFileTop-nano_",
             "WZ"    : "flatTreeFileTop-nano_"
             }


original_paths_dict = {"Higgs"  : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/HiggsZProduction/MC-CandidatesP8-Higgs-GenInfos-v5-ext1/180518_072651/",
                                          "/pnfs/desy.de/cms/tier2/store/user/pgunnell/HiggsZProduction/MC-CandidatesP8-Higgs-GenInfos-v4/180413_072857/"]),
                       "QCD"    : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/MC-CandidatesP8-QCD-GenInfos-v4/180413_073311/"]),
                       "Top"    : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MC-CandidatesP8-Top-GenInfos-v5/180503_120147/"]),
                       "DY"     : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/ZJetsToQQ_HT400to600_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-DY-GenInfos-v4/180504_114205/",
                                          "/pnfs/desy.de/cms/tier2/store/user/pgunnell/ZJetsToQQ_HT600to800_3j_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-DY-GenInfos-v4-bin600800HT/180504_185633/",
                                          "/pnfs/desy.de/cms/tier2/store/user/pgunnell/ZJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-DY-GenInfos-v4-bin800InfHT/180504_185003/"]),
                       "WJets"  : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/WJetsToQQ_HT400to600_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-W-GenInfos-v4-bin400600HT/180504_193010/",
                                          "/pnfs/desy.de/cms/tier2/store/user/pgunnell/WJetsToQQ_HT600to800_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-W-GenInfos-v4-bin600800HT/180504_190237/",
                                          "/pnfs/desy.de/cms/tier2/store/user/pgunnell/WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/MC-CandidatesP8-W-GenInfos-v4-bin800InfHT/180504_190341/"]),
                       "ZZ"     : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/ZZ_TuneCP5_13TeV-pythia8/MC-CandidatesP8-ZZ-GenInfos-v1/181119_153406/"]),
                       "WZ"     : sorted(["/pnfs/desy.de/cms/tier2/store/user/pgunnell/WZ_TuneCP5_13TeV-pythia8/MC-CandidatesP8-WZ-GenInfos-v1/181119_154113/"])
                       }


try:
    MakeOutDirs(bkgs, radii, branch_names_dict, out_path)
    files_dictionary = countOriginalFiles(bkgs,original_paths_dict)
    # print(files_dictionary)
except:
    files_dictionary = {'Top': (4419, ['0', '1090', '1384', '1385', '1387', '1500', '1577', '1628', '1786', '1808', '1811', '1812', '1843', '1856', '1870', '1871', '1962', '1975', '2002', '2013', '2015', '2016', '2055']),
                        'QCD': (1957, ['0', '286', '287', '288', '289', '290', '291', '700', '701', '702', '703', '704', '705', '706', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1154', '1155', '1156', '1157', '1455', '1456', '1457']),
                        'Higgs': (1574, ['0'])}
    # files_dictionary= {"Higgs":[1000], "QCD":[1000], "Top":[1000]}
    print("Default Dictionary: ")
    print(files_dictionary)
