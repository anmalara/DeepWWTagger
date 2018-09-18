import numpy as np
import time
import os
import os.path
import sys

from variables import *

def Sequential_var_selection(file_path, name_folder, info, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max):
    first = True
    for i in range(file_min, file_max):
        if (((i-file_min)*100./(file_max-file_min))%10==0): print("progress: "+str(int((i-file_min)*100./(file_max-file_min)))+" %")
        file_name = file_path+name_folder+bkg+"_"+radius+"/"+info+"_"+bkg+"_"+str(i)+".npy"
        if not os.path.isfile(file_name):
            continue
        jet_vars = np.load(file_name)
        jet_vars = jet_vars[(jet_vars[:,branch_names_dict["JetInfo"].index("jetPt")]>pt_min)*(jet_vars[:,branch_names_dict["JetInfo"].index("jetPt")]<pt_max)]
        if first:
            jet_info = jet_vars
            first = False
        else:
            jet_info = np.concatenate((jet_info,jet_vars))
    if not first:
        print jet_info.shape
        np.save(file_path+name_folder_output+"Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", jet_info)

def merge_file(file_path, name_folder_output, info, bkg, radius, file_min, file_max, pt_min, pt_max, step):
    first = True
    for i in range(file_min, file_max, step):
        file_name = file_path+name_folder_output+"Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(i)+"_"+str(i+step)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
        if not os.path.isfile(file_name):
            with open(file_path+name_folder_output+"/problem_"+bkg+radius+"_"+str(pt_min)+"_"+str(pt_max)+".txt","a") as F:
                F.write("Missing "+bkg+"-"+radius+"-"+str(pt_min)+"-"+str(pt_max)+" "+str(i)+"_"+str(i+step)+"\n")
            continue
        file = np.load(file_name)
        print file.shape
        if first:
            final = file
            first = False
        else:
            final = np.concatenate((final,file))
        temp_fold = file_path+name_folder_output+"save/"
        if not os.path.exists(temp_fold):
            os.makedirs(temp_fold)
        os.system("mv "+file_name+" "+temp_fold)
    if not first:
        print final.shape
        np.save(file_path+name_folder_output+"Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", final)


##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

file_path = out_path
name_folder = "input_varariables/NTuples_Tagger/"
name_folder_output = name_folder+"Sequential/"
if not os.path.exists(file_path+name_folder_output):
    os.makedirs(file_path+name_folder_output)

try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    bkg = sys.argv[3]
    radius = sys.argv[4]
    info = sys.argv[5]
    pt_min = int(sys.argv[6])
    pt_max = int(sys.argv[7])
    step = int(sys.argv[8])
    flag_merge = int(sys.argv[9])
except :
    file_min = 0
    file_max = 10
    bkg = "Higgs"
    radius = "AK8"
    info = "JetInfo"
    pt_min = 300
    pt_max = 500
    step = 10
    flag_merge = 0

# if name_variable=='norm':
#     name_variable=""

temp_fold = file_path+name_folder_output+"save/"
if not os.path.exists(temp_fold):
    os.makedirs(temp_fold)

if flag_merge:
    merge_file(file_path, name_folder_output, info, bkg, radius, file_min, file_max, pt_min, pt_max, step)
else:
    Sequential_var_selection(file_path, name_folder, info, bkg, name_folder_output, file_min, file_max, radius, pt_min, pt_max)
