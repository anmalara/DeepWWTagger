from root_numpy import root2array
from root_numpy import rec2array
import numpy as np
import math
import time
import os.path
import sys
import copy

from variables import *
#TODO: atleast_3d

def bubbleSort(matrix, col_max, px_index, py_index):
    for col in range(0, col_max-1):
        for i in range(col_max-1, col, -1):
            if math.sqrt(matrix[px_index][col]**2 + matrix[py_index][col]**2)< math.sqrt(matrix[px_index][i]**2 + matrix[py_index][i]**2):
                matrix[:,[col, i]] = matrix[:,[i, col]]


def mergeSort(matrix, col_max, px_index, py_index):
    if matrix.shape[1]>1:
        mid = matrix.shape[1]//2
        lefthalf  = copy.deepcopy(matrix[:, :mid])
        righthalf = copy.deepcopy(matrix[:, mid:])
        mergeSort(lefthalf, col_max, px_index, py_index)
        mergeSort(righthalf, col_max, px_index, py_index)
        i=0
        j=0
        k=0
        while i < lefthalf.shape[1] and j < righthalf.shape[1]:
            if math.sqrt(lefthalf[px_index][i]**2 + lefthalf[py_index][i]**2)> math.sqrt(righthalf[px_index][j]**2 + righthalf[py_index][j]**2):
                matrix[:,k]=lefthalf[:,i]
                i=i+1
            else:
                matrix[:,k]=righthalf[:,j]
                j=j+1
            k=k+1
        while i < lefthalf.shape[1]:
            matrix[:,k]=lefthalf[:,i]
            i=i+1
            k=k+1
        while j < righthalf.shape[1]:
            matrix[:,k]=righthalf[:,j]
            j=j+1
            k=k+1



def selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen):
    file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
    file = rec2array(file)
    if isGen:
        px_index = branch_names.index("GenCandPx")
        py_index = branch_names.index("GenCandPy")
    else:
        px_index = branch_names.index("CandPx")
        py_index = branch_names.index("CandPy")
    # print("len_file = "+str(len(file)))
    firstEvent = True
    for x in range(0,len(file)):
        col_max = 0
        firstColumn = True
        for y in range(0,len(branch_names)):
            col_max = max(col_max, file[x,y].shape[0])
            temp = file[x,y].reshape(1,file[x,y].shape[0])
            temp = temp.astype(variable_type)
            if firstColumn:
              info = temp
              firstColumn = False
            else:
              info = np.concatenate((info, temp))
        # info.shape = (len(branch_names), file[x,y].shape[0])
        # bubbleSort(info, col_max, px_index, py_index)
        mergeSort(info, col_max, px_index, py_index)
        if info.shape[1] >= ncand_tosave:
            temp_jet = info[:,:ncand_tosave]
        else:
            temp_jet = np.hstack((info, np.zeros((info.shape[0],ncand_tosave-info.shape[1]))))
        temp_jet=temp_jet.reshape(1,len(branch_names), ncand_tosave)
        if firstEvent:
            info_candidates = temp_jet
            firstEvent = False
        else:
            info_candidates = np.concatenate((info_candidates, temp_jet))
    # return numpy.ndarray whose shape is (n_events,branches.size(), ncand_tosave)
    return info_candidates

def selectBranches_Event(file_name, tree_name, branch_names, selection_cuts):
    file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
    # it needs 2 steps to a proper coversion into numpy.ndarray
    file = rec2array(file)
    file = file.astype(variable_type)
    # return numpy.ndarray whose shape is (n_events,branches.size())
    return file

def selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts):
    file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
    # it needs 2 steps to a proper coversion into numpy.ndarray
    file = rec2array(file)
    for x in range(0,len(file)):
        for y in range(0, len(branch_names)):
            if len(file[x,y])<1:
                file[x,y] = OutOfRangeValue
            else:
                file[x,y] = file[x,y][0]
    # convert from float32 to float16
    file = file.astype(variable_type)
    # return numpy.ndarray whose shape is (n_events,branches.size())
    return file

@timeit
def selectBranches(bkg, radius, file_min, file_max):
    tree_name = "boosted"+radius+"/events"
    for index, file in enumerate(original_paths_dict[bkg]):
        counter = 10000*index
        for i in range(file_min, file_max):
            if not i in files_dictionary[bkg]["elements"]:
                continue
            i -= counter
            if i<0 or i> 10000:
                continue
            subCounter = "000"+str(i/1000)+"/"
            file_name = file+subCounter+tree_dict[bkg]+str(i)+".root"
            if not os.path.exists(file_name):
                print file_name
                continue
            for info in branch_names_dict:
                branch_names = branch_names_dict[info]
                isGen = 0
                if "Gen" in info:
                    isGen = 1
                if "Event" in info:
                    array = selectBranches_Event(file_name, tree_name, branch_names, selection_cuts)
                if "Jet" in info:
                    array = selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts)
                if "Cand" in info:
                    array = selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen)
                array = array.astype(variable_type)
                outputfile = out_path+"input_varariables/NTuples_Tagger/"+bkg+"_"+radius+"/"+info+"/"+info+"_"+str(i+counter)
                print outputfile
                np.save(outputfile+".npy",array)
                time.sleep(1)
                del array




#############################
#                           #
#       MAIN Program        #
#                           #
#############################

try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    bkg = sys.argv[3]
    radius = sys.argv[4]
except:
    file_min = 20880
    file_max = 20980
    bkg = "WJets"
    radius = "AK15"

print file_min, file_max, bkg, radius

selectBranches(bkg, radius, file_min, file_max)
