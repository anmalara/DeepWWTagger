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


def plot_ROC_Curve(tpr, fpr, name_plot, name_fig):
    plt.cla()
    plt.figure()
    plt.semilogy([0, 1], [0, 1], 'k--')
    plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
    plt.grid(True, which='both')
    for i in range(0,len(tpr)):
        plt.semilogy(tpr[i], fpr[i], label=name_plot[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.001, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='best', shadow=True)
    plt.savefig(name_fig)


tpr_array = []
fpr_array = []
name_array = []

losses = []
accuracy = []
val_losses = []
val_accuracy = []


file_path = "/beegfs/desy/user/amalara/output_varariables/"

if not os.path.exists(file_path+"ROC/"):
    os.makedirs(file_path+"ROC/")

epoch = 100

pt_min=[300, 500]
pt_max=[500, 10000]

for radius in ["AK8", "AK15", "CA15"]:
    for index in range(len(pt_min)):
        for name_variable in ["", "gen_"]:
            tpr_array = []
            fpr_array = []
            name_array = []
            # for method in ["Sequential", "Lola", "JetImage"]:
            for method in ["JetImage"]:
                if "Sequential" in method:
                    epoch = 1
                if "Lola" in method:
                    epoch = 1
                if "JetImage" in method:
                    epoch = 1
                if "Sequential" in method:
                    variable_used=name_variable+"subjet"
                else:
                    variable_used=name_variable
                if not os.path.isfile(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy"):
                    continue
                fpr = np.load(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr = np.load(file_path+method+"/"+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                print file_path+method+"/"+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy"
                print fpr.shape, tpr.shape
                tpr_array.append(tpr)
                fpr_array.append(fpr)
                name_array.append(method+"-"+radius+"_var_"+variable_used+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index]))
            print file_path+"ROC/ROC_"+radius+"_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png"
            plot_ROC_Curve(tpr_array, fpr_array, name_array, file_path+"ROC/ROC_"+radius+"_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png")


for method in ["Sequential", "Lola", "JetImage"]:
    for index in range(len(pt_min)):
        for name_variable in ["", "gen_"]:
            tpr_array = []
            fpr_array = []
            name_array = []
            for radius in ["AK8", "AK15", "CA15"]:
                if "Sequential" in method:
                    epoch = 100
                if "Lola" in method:
                    epoch = 1
                if "JetImage" in method:
                    epoch = 1
                if "Sequential" in method:
                    variable_used=name_variable+"subjet"
                else:
                    variable_used=name_variable
                if not os.path.isfile(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy"):
                    continue
                fpr = np.load(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr = np.load(file_path+method+"/"+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr_array.append(tpr)
                fpr_array.append(fpr)
                name_array.append(method+"-"+radius+"_var_"+variable_used+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index]))
            print file_path+"ROC/ROC_"+method+"_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png"
            plot_ROC_Curve(tpr_array, fpr_array, name_array, file_path+"ROC/ROC_"+method+"_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png")


for method in ["Sequential", "Lola", "JetImage"]:
    for index in range(len(pt_min)):
        for radius in ["AK8", "AK15", "CA15"]:
            tpr_array = []
            fpr_array = []
            name_array = []
            for name_variable in ["", "gen_"]:
                if "Sequential" in method:
                    epoch = 100
                if "Lola" in method:
                    epoch = 1
                if "JetImage" in method:
                    epoch = 1
                if "Sequential" in method:
                    variable_used=name_variable+"subjet"
                else:
                    variable_used=name_variable
                if not os.path.isfile(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy"):
                    continue
                fpr = np.load(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr = np.load(file_path+method+"/"+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr_array.append(tpr)
                fpr_array.append(fpr)
                name_array.append(method+"-"+radius+"_var_"+variable_used+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index]))
            print file_path+"ROC/ROC_"+radius+"_"+method+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png"
            plot_ROC_Curve(tpr_array, fpr_array, name_array, file_path+"ROC/ROC_"+radius+"_"+method+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".png")

for method in ["Sequential", "Lola", "JetImage"]:
    for name_variable in ["", "gen_"]:
        for radius in ["AK8", "AK15", "CA15"]:
            tpr_array = []
            fpr_array = []
            name_array = []
            for index in range(len(pt_min)):
                if "Sequential" in method:
                    epoch = 100
                if "Lola" in method:
                    epoch = 1
                if "JetImage" in method:
                    epoch = 1
                if "Sequential" in method:
                    variable_used=name_variable+"subjet"
                else:
                    variable_used=name_variable
                if not os.path.isfile(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy"):
                    continue
                fpr = np.load(file_path+method+"/"+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr = np.load(file_path+method+"/"+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index])+".npy")
                tpr_array.append(tpr)
                fpr_array.append(fpr)
                name_array.append(method+"-"+radius+"_var_"+variable_used+"_pt_"+str(pt_min[index])+"_"+str(pt_max[index]))
            print file_path+"ROC/ROC_"+radius+"_"+method+"_var_"+variable_used+"_epoch_"+str(epoch)+".png"
            plot_ROC_Curve(tpr_array, fpr_array, name_array, file_path+"ROC/ROC_"+radius+"_"+method+"_var_"+variable_used+"_epoch_"+str(epoch)+".png")


print "end"
