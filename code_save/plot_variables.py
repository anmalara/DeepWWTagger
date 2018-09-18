from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def check_variables(name_variable, pt_max):
    bin = 100;
    ranges = None
    if name_variable == "jetMass" :
        bin = 100;
    elif name_variable == "jetMassSoftDrop" :
        ranges = (0,200)
    elif name_variable == "GenJetMass" :
        ranges = (0,200)
    elif name_variable == "GenJetPt" :
        if pt_max > 500:
            ranges = (500,2000)
    elif name_variable == "GenSoftDropMass" :
        if pt_max > 500:
            ranges = (0,500)
    elif name_variable == "jetMass" :
        if pt_max > 500:
            ranges = (0,600)
    elif name_variable == "jetPt" :
        if pt_max > 500:
            ranges = (500,2000)
    return bin,ranges

def plot_Sequential_Variables(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output):
    files = []
    for bkg in bkgs:
        print bkg
        file_name = file_path+name_folder+"Sequential_"+gen+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        file = np.load(file_name)
        print file.shape
        files.append(file)
    for name_variable in variables:
        print name_variable
        bin, ranges = check_variables(name_variable, pt_max)
        if name_variable == "jetBtag":
            continue
        if name_variable == "isBJetGen":
            continue
        plt.cla()
        plt.figure()
        for i in range(0,len(files)):
            x= files[i][:,variables.index(name_variable)]
            x= x.astype(float)
            plt.hist(x, bins=bin, range=ranges, density=None, histtype='bar', alpha=0.75, color=None, label=bkgs[i], normed=True)
        plt.grid(True, which='both')
        plt.xlabel(name_variable, position=(1,1))
        plt.ylabel('A.U.', position=(1,1))
        plt.title(name_variable+"_Sequential_"+gen+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max))
        plt.legend(loc='best', shadow=True)
        # plt.show()
        # time.sleep(5)
        file_name = name_variable+"_Sequential_"+gen+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)
        plt.savefig(name_folder_output+file_name+".png")
        plt.savefig(name_folder_output+file_name+".pdf")
        plt.close()


def plot_Sequential_Variables_2D(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output):
    files = []
    for bkg in bkgs:
        print bkg
        file_name = file_path+name_folder+"Sequential_"+gen+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        file = np.load(file_name)
        print file.shape
        files.append(file)
    for var_1 in range(0,len(variables)):
        if variables[var_1] == "jetBtag":
            continue
        if variables[var_1] == "isBJetGen":
            continue
        bin_1, range_1 = check_variables(variables[var_1], pt_max)
        for var_2 in range(var_1+1,len(variables)):
            if variables[var_2] == "jetBtag":
                continue
            if variables[var_2] == "isBJetGen":
                continue
            bin_2, range_2 = check_variables(variables[var_2], pt_max)
            plt.cla()
            plt.figure()
            plt.suptitle(variables[var_1]+"_"+variables[var_2]+"_Sequential_"+gen+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max))
            n_subplot=len(files)*100+11
            for i in range(0,len(files)):
                x = files[i][:,var_1]
                x = x.astype(float)
                y = files[i][:,var_2]
                y = y.astype(float)
                # plt.hist2d(x, y, bins=(bin_1,bin_2), range=(range_1,range_2), density=None, histtype='bar', alpha=0.75, color=None, label=bkgs[i], normed=True)
                # print [[range_1[0],range_1[1]],[range_2[0],range_2[1]]]
                plt.subplot(n_subplot+i)
                plt.hist2d(x, y, bins=(bin_1,bin_2), alpha=0.75, label=bkgs[i], normed=True, cmap=color[i], norm=LogNorm())
                plt.colorbar()
                plt.grid(True, which='both')
                plt.xlabel(variables[var_1], position=(1,1))
                plt.ylabel(variables[var_2], position=(1,1))
                plt.title(bkgs[i])
            plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
            # plt.show()
            # time.sleep(5)
            file_name = "2D_"+variables[var_1]+"_"+variables[var_2]+"_Sequential_"+gen+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)
            plt.savefig(name_folder_output+file_name+".png")
            plt.savefig(name_folder_output+file_name+".pdf")
            plt.close()


def plot_Lola_Variables(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output):
    files = []
    for bkg in bkgs:
        print bkg
        file_name = file_path+name_folder+"lola_"+gen+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        file = np.load(file_name)
        print file.shape
        files.append(file)
    for name_variable in variables:
        print name_variable
        bin, ranges = check_variables(name_variable, pt_max)
        if name_variable == "jetBtag":
            continue
        if name_variable == "isBJetGen":
            continue
        plt.cla()
        plt.figure()
        for i in range(0,len(files)):
            temp= files[i][:,variables.index(name_variable),: ]
            temp= temp.astype(float)
            x=[]
            for j in range(0, temp.shape[1]):
                try:
                    x = np.concatenate((x, temp[:,j]))
                except :
                    x = temp[:,j]
            plt.hist(x, bins=bin, range=ranges, density=None, histtype='bar', alpha=0.75, color=None, label=bkgs[i], normed=True)
        plt.grid(True, which='both')
        plt.xlabel(name_variable, position=(1,1))
        plt.ylabel('A.U.', position=(1,1))
        plt.title(name_variable+"_Lola_"+gen+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max))
        plt.legend(loc='best', shadow=True)
        # plt.show()
        # time.sleep(5)
        file_name = name_variable+"_Lola_"+gen+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)
        plt.savefig(name_folder_output+file_name+".png")
        plt.savefig(name_folder_output+file_name+".pdf")
        plt.close()


def plot_JetImage_Variables(file_path, name_folder, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output):
    files = []
    for bkg in bkgs:
        print bkg
        file_name = file_path+name_folder+"JetImage_input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        file = np.load(file_name)
        print file.shape
        files.append(file)
    for name_variable in variables:
        print name_variable
        index = variables.index(name_variable)
        plt.cla()
        plt.figure()
        plt.suptitle(name_variable+"_JetImage_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max))
        n_subplot=len(files)*100+11
        for i in range(0,len(files)):
            for j in range(0, files[i].shape[0]):
                try:
                    x = x + files[i][j,index,:,:]
                except:
                    x = files[i][j,index,:,:]
            x = x.astype(float)
            plt.subplot(n_subplot+i)
            # plt.hist2d(x, y, bins=(bin_1,bin_2), alpha=0.75, label=bkgs[i], normed=True, cmap=color[i], norm=LogNorm())
            plt.imshow(x, alpha=0.75, label=bkgs[i], norm=LogNorm(), cmap=color[i])
            plt.colorbar()
            plt.grid(True, which='both')
            plt.xlabel("eta", position=(1,1))
            plt.ylabel("phi", position=(1,1))
            plt.title(bkgs[i])
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        # plt.show()
        # time.sleep(5)
        file_name = name_variable+"_JetImage_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)
        plt.savefig(name_folder_output+file_name+".png")
        plt.savefig(name_folder_output+file_name+".pdf")
        plt.close()


branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_gen_jet = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]



file_path = "/beegfs/desy/user/amalara/input_varariables/NTuples_Tagger/"
name_folder = "Sequential/"
name_folder_output = "/beegfs/desy/user/amalara/output_varariables/plot_variables/Sequential/"
bkgs = ["Higgs", "QCD"]
# bkgs = ["Higgs"]
color = ["Blues", "Oranges", "Reds", "Greens"]
file_min = 0
file_max = 2000

radii=["AK8", "AK15", "CA15"]
input_variable = ["", "gen_"]
branch_names = [branch_names_jet, branch_names_gen_jet]
pts = [[300,500], [500,10000]]

print "Sequential"
for radius in radii:
    for i in range(len(input_variable)):
        gen = input_variable[i]
        variables = branch_names[i]
        for pt in pts:
            print radius, gen, pt
            pt_min = pt[0]
            pt_max = pt[1]
            # plot_Sequential_Variables(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output)
            # plot_Sequential_Variables_2D(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output)

name_folder = "Lola/"
name_folder_output = "/beegfs/desy/user/amalara/output_varariables/plot_variables/Lola/"
branch_names = [["CandEnergy", "CandPx", "CandPy", "CandPz"], ["GenCandEnergy", "GenCandPx", "GenCandPy", "GenCandPz"]]

print "Lola"
for radius in radii:
    for i in range(len(input_variable)):
        gen = input_variable[i]
        variables = branch_names[i]
        for pt in pts:
            print radius, gen, pt
            pt_min = pt[0]
            pt_max = pt[1]
            # plot_Lola_Variables(file_path, name_folder, gen, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output)

name_folder = "JetImage/"
name_folder_output = "/beegfs/desy/user/amalara/output_varariables/plot_variables/JetImage/"
branch_names = [["neutral_hadron", "charged_hadron", "pt"]]
file_max = 100

print "JetImage"
for radius in radii:
    for variables in branch_names:
        for pt in pts:
            print radius, pt
            pt_min = pt[0]
            pt_max = pt[1]
            plot_JetImage_Variables(file_path, name_folder, bkgs, radius, file_min, file_max, pt_min, pt_max, variables, name_folder_output)


name_folder_input = "/beegfs/desy/user/amalara/output_varariables/plot_variables/"
name_folder_output = "/home/amalara/DeepWWTagger/transfer/"
os.system("cp -r "+name_folder_input+" "+name_folder_output)
