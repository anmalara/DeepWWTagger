from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag"]#, "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_candidate = ["candEnergy", "candPx", "candPy", "candPz", "candPt", "candEta", "candPhi"]#, "candPdgId", "candMass", "candDXY", "candDZ", "candPuppiWeight"]
branch_names_gen_jet = ["GenJetpt", "GenJeteta", "GenJetphi", "GenJetmass", "GenJetenergy", "isBJetGen"]
branch_names_gen_cand = ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz", "GenCandPt", "GenCandEta", "GenCandPhi"]

def Sequential_var_selection(file_path, name_folder, name_variable, file_min, file_max, pt_min, pt_max):
    for i in range(file_min, file_max):
        file_name = file_path+name_folder+name_variable+str(i)+".npy"
        if (not os.path.isfile(file_name)):
            continue
        jet_vars = np.load(file_name)
        jet_vars = jet_vars[(jet_vars[:,branch_names_jet.index("jetPt")]>pt_min)*(jet_vars[:,branch_names_jet.index("jetPt")]<pt_max)]
        if i==1:
            jet_info = jet_vars
        else:
            jet_info = np.concatenate((jet_info,jet_vars))
    return jet_info



pt_min = 300
pt_max = 500
file_min = 1
file_max = 20

file_path = "/beegfs/desy/user/amalara/input_varariables/NTuples_Tagger/"
out_file_path = "/beegfs/desy/user/amalara/output_varariables/"
out_file_path = "/home/amalara/DeepWWTagger/transfer/"
radius = "AK8"
name_folder = "Higgs_"+radius+"/"
name_variable = "jet_var_Higgs_"
Higgs = Sequential_var_selection(file_path, name_folder, name_variable, file_min, file_max, pt_min, pt_max)
print Higgs.shape

for i in range(0,Higgs.shape[1]):
    print i, Higgs[:,i].shape, branch_names_jet[i]
    plt.cla()
    plt.hist(Higgs[:,i], bins='auto')
    plt.savefig(out_file_path+"test/"+branch_names_jet[i]+".png")



#
# def poisson(k,l):
#     return math.exp(-l)*math.pow(l,k)/math.factorial(k)
#
# count = 3
#
# for bkg in range(0,count+1):
#     sig = count - bkg
#     sum_1 = 0
#     for k in range(0,count+1):
#         sum_1 = sum_1 + poisson(k,sig+bkg)
#     sum_2 = 0
#     for k in range(0,count+1):
#         sum_2 = sum_2 + poisson(k,bkg)
#     print bkg, sig, sum_1/sum_2

#
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("Sequential/fpr_epoch_20_pt_300_500.npy")
# tpr = np.load("Sequential/tpr_epoch_20_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 20, 300 < pt < 500')
#
# fpr = np.load("Sequential/fpr_epoch_200_pt_300_500.npy")
# tpr = np.load("Sequential/tpr_epoch_200_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 200, 300 < pt < 500')
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_seq_300_500.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("Sequential/fpr_epoch_20_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_20_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 20, pt > 500')
#
# fpr = np.load("Sequential/fpr_epoch_200_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_200_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 200, pt > 500')
#
# fpr = np.load("Sequential/fpr_epoch_500_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_500_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 500, pt > 500')
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')
#
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_seq_500.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_seq.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("JetImage/fpr_epoch_20_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_20_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 20, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_50_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_50_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 50, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_100_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_100_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 100, 300 < pt < 500')
#
# # fpr = np.load("JetImage/fpr_epoch_200_pt_300_500.npy")
# # tpr = np.load("JetImage/tpr_epoch_200_pt_300_500.npy")
# # plt.semilogy(tpr, fpr, label='JetImage epochs = 200, 300 < pt < 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_JetImage_300_500.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("JetImage/fpr_epoch_20_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_20_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 20, pt > 500')
#
# fpr = np.load("JetImage/fpr_epoch_50_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_50_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 50, pt > 500')
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')
#
# fpr = np.load("JetImage/fpr_epoch_100_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_100_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 100, pt > 500')
#
# # fpr = np.load("JetImage/fpr_epoch_200_pt_500_10000.npy")
# # tpr = np.load("JetImage/tpr_epoch_200_pt_500_10000.npy")
# # plt.semilogy(tpr, fpr, label='JetImage epochs = 200, pt > 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_JetImage_500.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_100_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_100_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 100, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')
#
# fpr = np.load("JetImage/fpr_epoch_100_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_100_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 100, pt > 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='best', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN_JetImage.png")
#
# ################################################################################
# ################################################################################
#
# plt.cla()
# plt.figure()
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')
#
# fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')
#
# fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
# tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')
#
# plt.semilogy([0, 1], [0, 1], 'k--')
# plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
# plt.grid(True, which='both')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.05])
# plt.ylim([0.001, 1.05])
# plt.xlabel('True Positive Rate')
# plt.ylabel('False Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc='upper left', shadow=True)
# # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.savefig("transfer/NN.png")
