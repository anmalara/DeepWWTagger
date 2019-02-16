import os
import os.path
import sys
from glob import glob
import time
import math

import numpy as np
from root_numpy import root2array, rec2array, fill_hist
import ROOT

sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94X_v3/CMSSW_9_4_10/src/UHH2/PersonalCode/")
sys.path.append("/nfs/dust/cms/user/amalara/WorkingArea/UHH2_94X_v3/CMSSW_9_4_10/src/UHH2/BoostedHiggsToWW/Analysis/macros/")
from parallelise import *
from tdrstyle_all import *
from ModuleRunnerBase import ModuleRunnerBase


class TaggerBase(ModuleRunnerBase):
    def __init__(self, Sample="MC_HZ", Channel="muon", Collection="Puppi"):
        self.Base = ModuleRunnerBase()
        self.TreeName = "AnalysisTree"
        self.Module = "GenericCleaning"
        self.Mode = "MC"
        self.Channel = Channel
        self.Collection = Collection
        self.Sample = Sample
        self.FileStorageInput = self.Base.Path_STORAGE+"Analysis/"+self.Module+"_"+self.Mode+"/"+self.Collection+"/"+self.Channel+"channel"+"/"
        self.FileStorageOutput = self.Base.Path_STORAGE+"NeuralNetwork/input_varariables/EventTagger/"+self.Collection+"/"+self.Channel+"channel"+"/"
        self.File = self.FileStorageInput+self.Base.PrefixrootFile+self.Mode+"."+self.Sample+".root"
        self.MakeVars()
    def MakeVars(self):
        self.ObjectCollection = {"Event"    : "",
                                 "Jet"      : "slimmedJets.m_", #"updatedPatJetsSlimmedJetsPuppi.m_"
                                 "TopJet"   : "updatedPatJetsSlimmedJetsAK8_SoftDropPuppi.m_" if self.Collection=="Puppi" else "packedPatJetsAk8CHSJets_SoftDropCHS.m_",
                                 "Electron" : "slimmedElectronsUSER.m_",
                                 "Muon"     : "slimmedMuonsUSER.m_"}
        self.VarNameListsDict = {}
        self.VarNameListsDict["Event"] = ["event", "genInfo.m_weights", "weight_GLP", "weight_lumi", "weight_pu", "weight_pu_down", "weight_pu_up"]
        self.VarNameListsDict["Jet"] = ["pt", "eta", "phi", "energy", "numberOfDaughters", "jetArea",
         "neutralEmEnergyFraction", "neutralHadronEnergyFraction", "chargedEmEnergyFraction", "chargedHadronEnergyFraction", "muonEnergyFraction", "photonEnergyFraction",
         "chargedMultiplicity", "neutralMultiplicity", "muonMultiplicity", "electronMultiplicity", "photonMultiplicity",
         "puppiMultiplicity", "neutralPuppiMultiplicity", "neutralHadronPuppiMultiplicity", "photonPuppiMultiplicity", "HFHadronPuppiMultiplicity", "HFEMPuppiMultiplicity",
         "btag_combinedSecondaryVertex", "btag_combinedSecondaryVertexMVA", "btag_DeepCSV_probb", "btag_DeepCSV_probbb", "btag_BoostedDoubleSecondaryVertexAK8", "btag_BoostedDoubleSecondaryVertexCA15"]
        self.VarNameListsDict["TopJet"] = self.VarNameListsDict["Jet"] + ["subjets", "tau1", "tau2", "tau3", "tau4", "tau1_groomed", "tau2_groomed", "tau3_groomed", "tau4_groomed", "prunedmass", "softdropmass"]
        self.VarNameListsDict["Electron"] = ["pt", "eta", "phi", "energy", "charge"]
        self.VarNameListsDict["Muon"] = self.VarNameListsDict["Electron"]
        self.nObjects = 4
        self.VarNamesDict = {}
        self.Vars = {}
        for Objects in self.ObjectCollection:
            self.VarNamesDict[Objects] = [self.ObjectCollection[Objects]+var for var in self.VarNameListsDict[Objects]]
    def LoadObjectVars(self,Objects):
        print "Load ", Objects
        # Vars = root2array(filenames=self.File, treename=self.TreeName, branches=self.VarNamesDict[Objects], start=0 , stop=1000)
        Vars = root2array(filenames=self.File, treename=self.TreeName, branches=self.VarNamesDict[Objects])
        Vars = rec2array(Vars)
        print "Make ", Objects
        if Objects=="Event":
            for col, var in enumerate(Vars[0]):
                if (isinstance(var,np.ndarray)):
                    Vars[:,col] = np.array(map(lambda x: x[0], Vars[:,col]))
            self.Vars[Objects] = Vars
        else:
            VarList = []
            for n_jet in range(0,self.nObjects):
                for col, var in enumerate(Vars[0]):
                    VarList.append(np.expand_dims(np.array(map(lambda x: x[n_jet] if x.shape[0]>n_jet else 0, Vars[:,col])), axis=1))
            self.Vars[Objects] = np.concatenate(VarList,axis=1)
        print "Shape ", Objects, ":\t", self.Vars[Objects].shape
    def SaveVars(self):
        if not os.path.exists(self.FileStorageOutput+self.Sample+"/"):
            os.makedirs(self.FileStorageOutput+self.Sample+"/")
        for Objects in self.ObjectCollection:
            print "Save ", Objects
            np.save(self.FileStorageOutput+self.Sample+"/"+Objects+".npy", self.Vars[Objects])
    def LoadVars(self):
        self.Vars = {}
        for Objects in self.ObjectCollection:
            self.LoadObjectVars(Objects)
    def LoadSavedVars(self,path=""):
        if path=="":
            path = self.FileStorageOutput
        self.Vars = {}
        for Objects in self.ObjectCollection:
            print "Load ", Objects
            self.Vars[Objects] = np.load(path+self.Sample+"/"+Objects+".npy")




class EventTagger(TaggerBase):
    def __init__(self, Channel="muon", Collection="Puppi"):
        self.SamplesNames = {"MC_HZ":0, "MC_DYJets":1, "MC_TTbar":2, "MC_WZ":3, "MC_ZZ":4}
        self.TaggerBaseDict = {}
        for Sample in self.SamplesNames:
            self.TaggerBaseDict[Sample] = TaggerBase(Sample,Channel,Collection)
    def CreateVars(self):
        for Sample in self.SamplesNames:
            print "CreateVars:", Sample
            self.TaggerBaseDict[Sample].LoadVars()
            self.TaggerBaseDict[Sample].SaveVars()
    def LoadVars(self,path=""):
        for Sample in self.SamplesNames:
            print "LoadVars:", Sample
            self.TaggerBaseDict[Sample].LoadSavedVars(path)
    def PlotVars(self):
        colors = {"MC_HZ" :     ROOT.kRed+1,
                  "MC_DYJets":  ROOT.kGreen-2,
                  "MC_TTbar":   ROOT.kBlue,
                  "MC_WZ":      ROOT.kOrange,
                  "MC_ZZ":      ROOT.kViolet-3}
        def SetRanges(var):
            min = 0
            max = 100
            bin = 100
            if "Fraction" in var or "tau" in var or "weight" in var:
                max = 1
            if "btag" in var:
                min = -1
                max = 1
            if "eta" in var or "phi" in var:
                min = -math.pi
                max = math.pi
            if "pt" in var or "energy" in var:
                max = 1000
            if "mass" in var:
                max = 200
            if "charge" in var:
                min = -2
                max = 2
            return min, max, bin
        SampleRef = "MC_HZ"
        self.pathPlots = self.TaggerBaseDict[SampleRef].FileStorageOutput+"plots/"
        if not os.path.exists(self.pathPlots):
            os.makedirs(self.pathPlots)
        canvases = []
        ROOT.gROOT.SetBatch(ROOT.kFALSE)
        ROOT.gROOT.SetBatch(ROOT.kTRUE)
        ROOT.gStyle.SetOptStat(1)
        ROOT.gStyle.SetOptStat(0)
        for Objects in self.TaggerBaseDict[SampleRef].ObjectCollection:
            print Objects
            for index, var in enumerate(self.TaggerBaseDict[SampleRef].VarNameListsDict[Objects]):
                min, max, bin = SetRanges(var)
                c_ = tdrCanvas(Objects+var, min, max, 1.e-4, 1.e02, var, "A.U.")
                c_.SetLogy(1)
                leg = tdrLeg(0.50,0.70,0.9,0.9, 0.035)
                histos = []
                for Sample in self.SamplesNames:
                    h_ = ROOT.TH1F(Objects+var+Sample, "; "+var+"; A.U.", bin, min, max)
                    histos.append(h_)
                    fill_hist(h_, self.TaggerBaseDict[Sample].Vars[Objects][:,index])
                    if (h_.Integral()!=0):
                        h_.Scale(1./h_.Integral())
                    tdrDraw(h_, "same", ROOT.kFullCircle, colors[Sample], 1, colors[Sample], 0, colors[Sample])
                    leg.AddEntry(h_, Sample ,"lep")
                leg.SetLineColorAlpha(1, 0.7)
                leg.Draw("same")
                c_.SaveAs(self.pathPlots+Objects+var+".pdf", "pdf");
                canvases.append(c_)
                time.sleep(2)
        return canvases


#
# for Channel in ["muon","electron"]:
#     for Collection in ["CHS","Puppi"]:
#         EventClassifier = EventTagger(Channel, Collection)
#         EventClassifier.CreateVars()
#         # EventClassifier.LoadVars()
#         EventClassifier.PlotVars()
