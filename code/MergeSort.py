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

def shellSort(alist):
    sublistcount = len(alist)//2
    while sublistcount > 0:
      for startposition in range(sublistcount):
        gapInsertionSort(alist,startposition,sublistcount)
      print("After increments of size",sublistcount,"The list is",alist)
      sublistcount = sublistcount // 2

def gapInsertionSort(alist,start,gap):
    print start,gap, range(start+gap,len(alist),gap)
    print "list first ", alist
    for i in range(start+gap,len(alist),gap):
        currentvalue = alist[i]
        check = alist[i]
        position = i
        print i, currentvalue, position-gap, alist[position-gap]
        while position>=gap and alist[position-gap]>check:
            print "list swap1 ", alist
            alist[position]=alist[position-gap]
            position = position-gap
            print "list wwap2 ", alist
        alist[position]=currentvalue
        print "list after ", alist

alist = [54,26,93,17,77,31,44,55,20]
shellSort(alist)
print(alist)



def bubbleSort(matrix, col_max, px_index, py_index):
  for col in range(0, col_max-1):
    for i in range(col_max-1, col, -1):
      if math.sqrt(matrix[px_index][col]**2 + matrix[py_index][col]**2)< math.sqrt(matrix[px_index][i]**2 + matrix[py_index][i]**2):
        matrix[:,[col, i]] = matrix[:,[i, col]]

array = selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen)

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

array1 = selectBranches_Candidate2(file_name, tree_name, branch_names, selection_cuts, isGen)


def shellSort(matrix, col_max, px_index, py_index):
  gap = matrix.shape[1]//2
  # gap = matrix.shape[1]//min(1000,matrix.shape[1])
  # gap = 1
  while gap > 0:
    for startposition in range(gap):
      gapInsertionSort(matrix,startposition,gap, col_max, px_index, py_index)
    gap = gap // 2

def gapInsertionSort(matrix,start,gap, col_max, px_index, py_index):
  for i in range(start+gap,matrix.shape[1],gap):
    currentvalue = copy.deepcopy(matrix[:,i])
    check = copy.deepcopy(math.sqrt(matrix[px_index,i]**2 + matrix[py_index,i]**2))
    position = i
    while position>=gap and math.sqrt(matrix[px_index][position-gap]**2 + matrix[py_index][position-gap]**2)<check:
      matrix[:,position]=matrix[:,position-gap]
      position = position-gap
    matrix[:,position]=currentvalue

array2 = selectBranches_Candidate3(file_name, tree_name, branch_names, selection_cuts, isGen)
a= array[:,:,:]-array2[:,:,:]
a[a!=0]

@timeit
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
  # for x in range(0,len(file)):
  for x in range(0,100):
  # for x in range(0,100):
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
    # shellSort(info, col_max, px_index, py_index)
    if info.shape[1] >= ncand:
      temp_jet = info[:,:ncand]
    else:
      temp_jet = np.hstack((info, np.zeros((info.shape[0],ncand-info.shape[1]))))
    temp_jet=temp_jet.reshape(1,len(branch_names), ncand)
    if firstEvent:
      info_candidates = temp_jet
      firstEvent = False
    else:
      info_candidates = np.concatenate((info_candidates, temp_jet))
  # return numpy.ndarray whose shape is (n_events,branches.size(), ncand)
  return info_candidates


@timeit
def selectBranches_Candidate2(file_name, tree_name, branch_names, selection_cuts, isGen):
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
  # for x in range(0,len(file)):
  for x in range(0,100):
    col_max = 0
    firstColumn = True
    for y in range(0,len(branch_names)):
      col_max = max(col_max, file[x,y].shape[0])
      temp = np.expand_dims(file[x,y], axis=0).astype(variable_type)
      if firstColumn:
        info = temp
        firstColumn = False
      else:
        info = np.concatenate((info, temp))
    # info.shape = (len(branch_names), file[x,y].shape[0])
    mergeSort(info, col_max, px_index, py_index)
    if info.shape[1] >= ncand:
      temp_jet = info[:,:ncand]
    else:
      temp_jet = np.hstack((info, np.zeros((info.shape[0],ncand-info.shape[1]))))
    temp_jet=np.expand_dims(temp_jet, axis=0).astype(variable_type)
    if firstEvent:
      info_candidates = temp_jet
      firstEvent = False
    else:
      info_candidates = np.concatenate((info_candidates, temp_jet))
  # return numpy.ndarray whose shape is (n_events,branches.size(), ncand)
  return info_candidates


@timeit
def selectBranches_Candidate3(file_name, tree_name, branch_names, selection_cuts, isGen):
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
  # for x in range(0,len(file)):
  for x in range(0,100):
    col_max = 0
    firstColumn = True
    for y in range(0,len(branch_names)):
      col_max = max(col_max, file[x,y].shape[0])
      temp = np.expand_dims(file[x,y], axis=0).astype(variable_type)
      if firstColumn:
        info = temp
        firstColumn = False
      else:
        info = np.concatenate((info, temp))
    # info.shape = (len(branch_names), file[x,y].shape[0])
    shellSort(info, col_max, px_index, py_index)
    if info.shape[1] >= ncand:
      temp_jet = info[:,:ncand]
    else:
      temp_jet = np.hstack((info, np.zeros((info.shape[0],ncand-info.shape[1]))))
    temp_jet=np.expand_dims(temp_jet, axis=0).astype(variable_type)
    if firstEvent:
      info_candidates = temp_jet
      firstEvent = False
    else:
      info_candidates = np.concatenate((info_candidates, temp_jet))
  # return numpy.ndarray whose shape is (n_events,branches.size(), ncand)
  return info_candidates


@timeit
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
def selectBranches(name_folder_input, name_variable, bkg, name_folder_output, file_min, file_max, radius):
  for i in range(file_min, file_max):
    subCounter = "000"+str(i/1000)+"/"
    if i>10000:
      raise Exception("ERROR IN INPUT FILES")
    file_name = name_folder_input+subCounter+name_variable+str(i)+".root"
    if not os.path.exists(file_name):
      print file_name
      continue
    for info in branch_names_dict:
      if "Event" in info or "Extra" in info:
        continue
      branch_names = branch_names_dict[info]
      isGen = 0
      if "Gen" in info:
        isGen = 1
      if "Jet" in info:
        array = selectBranches_Jet(file_name, tree_name, branch_names, selection_cuts)
      if "Cand" in info:
        # array = selectBranches_Candidate(file_name, tree_name, branch_names, selection_cuts, isGen)
        array = selectBranches_Candidate2(file_name, tree_name, branch_names, selection_cuts, isGen)
      array = array.astype(variable_type)
      print array.shape
      outputfile = name_folder_output+radius+"/"+info+"_"+bkg+"_"+str(i)
      np.save(outputfile+".npy",array)
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
except :
  file_min = 0
  file_max = 10
  bkg = "Higgs"
  radius = "AK8"

name_folder_input = original_paths_dict[bkg]

name_variable = "flatTreeFile"+bkg+"-nano_"
tree_name = "boosted"+radius+"/events"
name_folder_output = out_path+"input_varariables/NTuples_Tagger/"+bkg+"_"


file_min = 10
file_max = 20
bkg = "Higgs"
radius = "AK15"

i= 10

subCounter = "000"+str(i/1000)+"/"
branch_names = branch_names_dict["CandInfo"]
isGen = 0
file_name = name_folder_input+subCounter+name_variable+str(i)+".root"



print file_min, file_max, bkg, radius

if not os.path.exists(name_folder_output+radius):
  os.makedirs(name_folder_output+radius)

print("process : "+bkg+"_"+radius)
selectBranches(name_folder_input, name_variable, bkg, name_folder_output, file_min, file_max, radius)
