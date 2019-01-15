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

# colors = [kBlack, kRed+1, kBlue-4, kGreen-2, kOrange, kMagenta, kViolet-3, kCyan, kSpring, kTeal, kYellow+1, kPink+10, kAzure+7, kAzure+1, kRed+3, kGray]

colors = {"Higgs": kRed+1, "QCD": kBlue-4, "Top": kGreen-2, "DY": kOrange, "WJets": kMagenta, "WZ": kViolet-3, "ZZ": kCyan}

def CreateHisto(name, Nbins, bin_min, bin_max, array_):
    array_ = array_.astype(np.float64)
    if len(array_.shape)==3:
        array = np.sum(array_, axis=0, dtype=np.float64)
        print "prima ", np.sum(array_, dtype=np.float64)
        array_ /= np.sum(array_, dtype=np.float64)
        print "dopo ", np.sum(array_, dtype=np.float64)
        h = TH2F( name, name, Nbins, bin_min, bin_max, Nbins, bin_min, bin_max,)
        for i in range(0,Nbins):
            for j in range(0,Nbins):
                h.SetBinContent(i+1,j+1,array[i,j])
    else:
        h = TH1F( name, name, Nbins, bin_min, bin_max)
        fill_hist(h, array_)
    return h

@timeit
def CreateDictHistos():
    HistoDict = {}
    for bkg in bkgs:
        dict_radius = {}
        for radius in radii:
            dict_pt = {}
            for index_pt, pt_min in enumerate(pts_min):
                pt_max = pts_max[index_pt]
                dict_info = {}
                for info in branch_names_list:
                    dict_var = {}
                    vector = []
                    sum = 0
                    for file_index in files_dictionary[bkg]["elements"]:
                        file_name = out_path+"input_varariables/NTuples_Tagger/Inputs/"+bkg+"_"+radius+"/"+info+"/file_"+str(file_index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
                        file_ = np.load(file_name)
                        if len(file_.shape)>0:
                            if len(file_.shape)==3:
                                if file_.shape[2]!=100:
                                    file_ = file_[:,:,:ncand_tosave]
                                if info == "JetImage":
                                    continue
                            vector.append(file_)
                            sum += len(file_)
                            if sum > max_file:
                                break
                    if len(vector)>0:
                        vector = np.concatenate(vector)
                        for index_variable, variable in enumerate(Info_dict[info]):
                            if len(vector.shape)==2:
                                array_ = vector[:,index_variable]
                            if len(vector.shape)==3:
                                array_ = np.concatenate(vector, axis=1).swapaxes(0,1)[:,index_variable]
                            if len(vector.shape)==4:
                                array_ = vector[:,index_variable,:,:]
                            plotInfo = getplotInfo(variable, radius, pt_min)
                            name = bkg+"_"+radius+"_"+str(pt_min)+"_"+variable
                            dict_var[variable] = {"histo": CreateHisto(name, plotInfo["Nbins"], plotInfo["bin_min"], plotInfo["bin_max"], array_), "plotInfo": plotInfo}
                    dict_info[info] = dict_var
                dict_pt[str(pt_min)] = dict_info
            dict_radius[radius] = dict_pt
        HistoDict[bkg] = dict_radius
    return HistoDict

def textmatch(variable, texts):
    found = False
    for text in texts:
        found = found or text.lower() in variable.lower()
    return found

def getplotInfo(variable="Pt", radius="AK8", pt = 300):
    a = ["jetPt", "", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4", "ncandidates", "CandEnergy", "CandPx", "CandPy", "CandPz", "CandPt", "CandEta", "CandPhi", "CandPdgId", "CandMass", "CandDXY", "CandDZ", "CandPuppiWeight"]
    plotInfo = {"Nbins": 100,  "bin_min": 0,  "bin_max": 1000, "min": 0.0001, "max": 1.0,  "isLogx": 0,  "isLogy": 0}
    if textmatch(variable, ["jetPt","jetEnergy"]):
        plotInfo["bin_min"] = 250 if pt == 300 else 300
        plotInfo["bin_max"] = 600 if pt == 300 else 5000
    if textmatch(variable, ["jetMass"]):
        plotInfo["bin_max"] = 300
        plotInfo["max"] = 0.15
    if textmatch(variable, ["ncandidates"]):
        plotInfo["bin_max"] = 200
    if textmatch(variable, ["jetEta","jetPhi"]):
        plotInfo["bin_min"] = -PI
        plotInfo["bin_max"] = +PI
        plotInfo["max"] = 0.05
    if textmatch(variable, ["jetTau", "jetBtag"]):
        plotInfo["bin_max"] = 1.05
        plotInfo["max"] = 0.35
    if textmatch(variable, ["jetBtag"]):
        plotInfo["max"] = 0.1
    if textmatch(variable, ["jetPt","jetEnergy","ncandidates", "CandPt","candEnergy","CandEta", "CandPhi","CandPx","CandPy","CandPz","CandMass","CandDXY","CandDZ","CandPuppiWeight", "CandPdgId"]):
        plotInfo["isLogy"] = 1
    if textmatch(variable, ["CandDXY", "CandDZ"]):
        plotInfo["bin_min"] = -0.4
        plotInfo["bin_max"] = +0.4
    if textmatch(variable, ["CandPx", "CandPy", "CandPz"]):
        plotInfo["bin_min"] = -1000
    if textmatch(variable, ["CandEta", "CandPhi"]):
        plotInfo["bin_min"] = -1.6
        plotInfo["bin_max"] = +1.6
    if textmatch(variable, ["CandPdgId"]):
        plotInfo["bin_min"] = -250
        plotInfo["bin_max"] = +250
        plotInfo["max"] = 10
    if textmatch(variable, ["CandMass"]):
        plotInfo["bin_min"] = -0.1
        plotInfo["bin_max"] = +0.4
    if textmatch(variable, ["CandPuppiWeight"]):
        plotInfo["bin_min"] = -0.05
        plotInfo["bin_max"] =  1.05
    if textmatch(variable, ["SumCandMass"]):
        plotInfo["isLogy"] = 0
        plotInfo["bin_max"] = 300
        plotInfo["max"] = 0.15
    if textmatch(variable, ["SumCandPx", "SumCandPy", "SumCandPz"]):
        plotInfo["bin_min"] = -3000
        plotInfo["bin_max"] =  3000
    if textmatch(variable, ["jetTau21", "jetTau31", "jetTau41", "jetTau32", "jetTau42", "jetTau43"]):
        plotInfo["max"] = 0.15
    if textmatch(variable, ["SumCandPt", "SumCandEnergy"]):
        plotInfo["bin_min"] = 250 if pt == 300 else 300
        plotInfo["bin_max"] = 600 if pt == 300 else 5000
    if textmatch(variable, ["NeutralHadron", "ChargedHadron", "PtPf"]):
        Radius = 0.8 if "8" in radius else 1.5
        plotInfo["Nbins"] = int(2*Radius/step_eta)
        plotInfo["bin_min"] = -Radius
        plotInfo["bin_max"] = +Radius
    return plotInfo

def SaveCanvas(c, name):
    c.Print(name+".pdf")
    c.Print(name+".png")
    c.Print(name+".root")

@timeit
def plotJetVariables():
    HistoDict = CreateDictHistos()
    for radius in radii:
        for pt_min in pts_min:
            for info in branch_names_list:
                output_path = out_path+"plot/"+radius+"/pt_"+str(pt_min)+"/"+info+"/"
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                for variable in Info_dict[info]:
                    plotInfo = getplotInfo(variable, radius, pt_min)
                    if info == "JetImage":
                        for bkg in bkgs:
                            h = HistoDict[bkg][radius][str(pt_min)][info][variable]["histo"]
                            if isinstance(h, TH2F):
                                c = tdrCanvas(variable+radius+str(pt_min), plotInfo["bin_min"], plotInfo["bin_max"], plotInfo["min"], plotInfo["max"], variable, "A.U.", square=kSquare, iPeriod=0, iPos=11, extraText_="Simulation")
                                gStyle.SetOptStat(0)
                                c.SetRightMargin(1)
                                c.SetLogz(1)
                                h.Draw("colz")
                                # h.GetListOfFunctions().FindObject("palette").SetX2NDC(0.94)
                                # c.Modified()
                                # c.Update()
                                # time.sleep(3)
                                SaveCanvas(c, output_path+variable+bkg)
                    else:
                        c = tdrCanvas(variable+radius+str(pt_min), plotInfo["bin_min"], plotInfo["bin_max"], plotInfo["min"], plotInfo["max"], variable, "A.U.", square=kRectangular, iPeriod=0, iPos=11, extraText_="Simulation")
                        c.SetLogx(plotInfo["isLogx"])
                        c.SetLogy(plotInfo["isLogy"])
                        leg = tdrLeg(0.6, 0.7, 0.9, 0.9, textSize=0.025)
                        tdrHeader(leg, variable)
                        for bkg in bkgs:
                            h = HistoDict[bkg][radius][str(pt_min)][info][variable]["histo"]
                            if isinstance(h, TH1F):
                                h.SetLineWidth(3)
                                if h.Integral()>0:
                                    h.Scale(1./h.Integral())
                                tdrDraw(h, "hist", mcolor=colors[bkg], lcolor=colors[bkg], fstyle=0, fcolor=colors[bkg])
                                leg.AddEntry(h, bkg + ", Entries: "+str(round(float(h.GetEntries())/1000000,3))+" M","l")
                        leg.Draw()
                        time.sleep(1)
                        SaveCanvas(c, output_path+variable)



max_file = 1000000
bkgs = ["Higgs", "QCD", "Top", "DY", "WJets"]

radii = ["AK15","AK8"]

branch_names_list = ["JetInfo", "CandInfo"]
branch_names_list = ["JetInfo", "CandInfo", "JetImage", "JetVariables"]
branch_names_list = ["JetVariables"]
# branch_names_list = ["JetInfo", "CandInfo", "JetVariables"]
#
# radius = "AK8"
# pt_min = "500"
# info = "CandInfo"
#
# variable = "jetPt"

plotJetVariables()
