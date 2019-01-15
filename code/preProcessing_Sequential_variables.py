import numpy as np
import time
import os
import os.path
import sys
import glob
from math import sqrt, fabs

from variables import *
from Prepocessing import *

def CreateJetVariables(Vars):
    for var in Vars:
        if "CandInfo" != var:
            continue
        for event in range(Vars[var].shape[0]):
            px, py, pz, pt, E  = 0, 0, 0, 0, 0
            for cand in range(Vars[var].shape[2]):
                px_ = Vars[var][event,CandPx_index,cand]
                py_ = Vars[var][event,CandPy_index,cand]
                pt_ = Vars[var][event,CandPt_index,cand]
                print sqrt(fabs(pt*pt-px*px-py*py)), pt, cand
                px += Vars[var][event,CandPx_index,cand]
                py += Vars[var][event,CandPy_index,cand]
                pz += Vars[var][event,CandPz_index,cand]
                pt += Vars[var][event,CandPt_index,cand]
                E  += Vars[var][event,CandEnergy_index,cand]
            print sqrt(pt*pt-px*px-py*py)/pt, sqrt(fabs(E*E-pz*pz-pt*pt)), Vars["JetInfo"][event,jetMassSoftDrop_index],Vars["JetInfo"][event,jetMass_index], event,var




@timeit
def Sequential_var_selection(folder_input, folder_output, file_min, file_max, bkg, radius, pt_min, pt_max):
    for i in range(file_min, file_max):
        infos = {}
        for info in branch_names_dict:
            infos[info]= []
        Vars = {}
        for info in infos:
            file_name = folder_input+bkg+"_"+radius+"/"+info+"/"+info+"_"+str(i)+".npy"
            if not os.path.isfile(file_name):
                with open(folder_output+"problem_"+bkg+"_"+radius+".txt","a") as F:
                    F.write("Missing "+bkg+"_"+radius+"_"+info+"_"+str(pt_min)+"_"+str(pt_max)+" "+str(i)+"\n")
                continue
            Vars[info] = np.load(file_name)
        preprocessing_pt_selection(Vars, "JetInfo", pt_min, pt_max)
        for i in Vars.keys(): print i, Vars[i].shape
        for i in Vars.keys(): print Vars[i].shape
        for info in infos:
            infos[info].append(Vars[info])
        for info in infos:
            infos[info] = np.concatenate(infos[info])
            infos[info] = infos[info].astype(variable_type)
            name = folder_output+info+"/Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
            print infos[info].shape, name
            np.save(name, infos[info])

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################


try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    bkg = sys.argv[3]
    radius = sys.argv[4]
    pt_min = int(sys.argv[5])
    pt_max = int(sys.argv[6])
except Exception as e:
    file_min = 10
    file_max = 20
    bkg = "Higgs"
    radius = "AK8"
    pt_min = 300
    pt_max = 500

folder_input  = out_path+"input_varariables/NTuples_Tagger/"
folder_output = out_path+"input_varariables/NTuples_Tagger/Sequential/"+bkg+"_"+radius+"/"


Sequential_var_selection(folder_input, folder_output, file_min, file_max, bkg, radius, pt_min, pt_max)
