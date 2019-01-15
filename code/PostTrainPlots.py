import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from UtilsForTraining import *
from variables import *


@timeit
def InputShape(sample_names=["Higgs","QCD", "Top"],max_size=500000):
    data = []
    labels = []
    for i, sample_name in enumerate(sample_names):
        subcounter = 0
        sample = []
        for index in files_dictionary[sample_name]["elements"]:
            temp = np.load(inputFolder+sample_name+"_"+radius+"/JetImage_"+info+"_"+sample_name+"_"+radius+"_file_"+str(index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", encoding = "bytes")
            if len(temp.shape)<4:
                continue
            temp = temp[~np.isinf(temp).any(axis=(1,2,3))]
            sample.append(temp)
            subcounter += temp.shape[0]
            if subcounter > max_size and max_size != -1:
                break
        sample = np.concatenate(sample)
        print(sample_name, sample.shape)
        label = np.ones(sample.shape[0],dtype=int)*i
        data.append(sample)
        labels.append(label)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, random_state=42, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test



info = "JetInfo"
radius = "AK15"
pt_min = 500
pt_max = 10000
name_model = "bestmodel_epoch003_acc0.87.h5"

sample_names = ["Higgs", "Top", "QCD"]
filePath = "/beegfs/desy/user/amalara/"


files_dictionary = {}
for bkg in bkgs:
    pt_name = "_pt_"+str(pt_min)+"_"+str(pt_max)
    outputdir = filePath+"input_varariables/NTuples_Tagger/JetImage/"+bkg+"_"+radius+"/JetImage_"+info+"_"+bkg+"_"+radius
    element = []
    for file_ in glob(outputdir+"*"+pt_name+".npy"):
        file_ = file_[:-4].replace(pt_name,"")
        file_ = file_[file_.rfind("_")+1:]
        element.append(file_)
    files_dictionary[bkg] = {"elements" : sorted((element))}


inputFolder = filePath+"input_varariables/NTuples_Tagger/JetImage/"
modelName = "model_"+info+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+"/"
outputFolder = filePath+"output_varariables/JetImage/"+modelName
modelpath = outputFolder+"models/"

model = load_model(modelpath+name_model)

data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names,max_size=-1)

100.*len(labels_test[labels_test[:,0]==1][:,0])/len(labels_test[:,0])

predictions = model.predict(data_test)

# plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name=modelpath+"Roc")

t_ = 1
bkg = 0
fpr, tpr, thr = roc_curve(labels_test[:,bkg], predictions[:,bkg])
index_01   = (np.abs(fpr - 0.1)).argmin()
index_001  = (np.abs(fpr - 0.01)).argmin()
index_0001 = (np.abs(fpr - 0.001)).argmin()
plt.cla()
plt.xticks( np.arange(0.1,1.1,0.1) )
plt.grid(True, which='both')
plt.axvline(x=tpr[index_01])
plt.axvline(x=tpr[index_001])
plt.axvline(x=tpr[index_0001])
plt.semilogy(tpr, fpr)
plt.show()

for bkg1 in [0,1,2]:
    plt.cla()
    plt.hist(predictions[labels_test[:,bkg]==t_][:,bkg1], alpha = 0.5, label=str(t_)+sample_names[bkg]+sample_names[bkg1], bins=100)
    plt.axvline(x=thr[index_01])
    plt.axvline(x=thr[index_001])
    plt.axvline(x=thr[index_0001])
    plt.legend(loc='best', shadow=True)
    plt.show()

quit()




























@timeit
def loadVariables(bkg):
    temp = inputFolder+"Sequential_"+info+"_"+bkg+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
    return np.load(temp, encoding = "bytes")

@timeit
def InputShape(sample_names=["Higgs","QCD", "Top"],max_size=500000):
    data = []
    labels = []
    for i, sample_name in enumerate(sample_names):
        sample = loadVariables(sample_name)
        sample = sample[~np.isinf(sample).any(axis=1)]
        print(sample_name, sample.shape)
        sample = sample[:max_size,:]
        print(sample_name, sample.shape)
        label = np.ones(sample.shape[0],dtype=int)*i
        data.append(sample)
        labels.append(label)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, random_state=42, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test

@timeit
def Normalization(X_standard, list_X):
    def FindScaler(name):
        if name=="jetPt": return "MinMaxScaler"
        elif name=="jetEta": return "StandardScaler"
        elif name=="jetPhi": return "MinMaxScaler"
        elif name=="jetMass": return "MinMaxScaler"
        elif name=="jetEnergy": return "MinMaxScaler"
        elif name=="jetBtag": return "MinMaxScaler"
        elif name=="jetMassSoftDrop": return "MinMaxScaler"
        elif name=="jetTau1": return "MinMaxScaler"
        elif name=="jetTau2": return "MinMaxScaler"
        elif name=="jetTau3": return "MinMaxScaler"
        elif name=="jetTau4": return "MinMaxScaler"
        elif name=="ncandidates": return "StandardScalerNoMean"
        else: return "MinMaxScaler"
    for nameBranch in branch_names_dict[info]:
        if nameBranch=="jetBtag":
            continue
        indexBranch = branch_names_dict[info].index(nameBranch)
        col_standard = X_standard[:,(indexBranch,)]
        if nameBranch == "jetMassSoftDrop" or nameBranch == "jetBtag":
            col_standard = col_standard[col_standard[:,0]>0,:]
        NameScaler = FindScaler(nameBranch)
        if NameScaler == "StandardScaler":
            scaler = preprocessing.StandardScaler().fit(col_standard)
        if NameScaler == "MinMaxScaler":
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(col_standard)
        if NameScaler == "StandardScalerNoMean":
            scaler = preprocessing.StandardScaler(with_mean=False).fit(col_standard)
        for X in list_X:
            X[:,(indexBranch,)] = scaler.transform(X[:,(indexBranch,)])






info = "JetInfo"
radius = "AK8"
pt_min = 500
pt_max = 10000
name_model = "bestmodel_epoch051_acc0.91.h5"

sample_names = ["Higgs", "Top", "QCD"]

filePath = "/beegfs/desy/user/amalara/"
inputFolder = filePath+"input_varariables/NTuples_Tagger/Sequential/"
modelName = "model_"+info+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+"/"
outputFolder = filePath+"output_varariables/Sequential/"+modelName
modelpath = outputFolder+"models/"

model = load_model(modelpath+name_model)


isSubset = True

isGen = 0
if "Gen" in info:
    isGen = 1

data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names,max_size=-1)

Normalization(data_train,[data_train,data_val,data_test])


if isSubset:
    if isGen:
        subset = (branch_names_dict["GenJetInfo"].index("GenSoftDropMass"), branch_names_dict["GenJetInfo"].index("GenJetTau1"), branch_names_dict["GenJetInfo"].index("GenJetTau2"), branch_names_dict["GenJetInfo"].index("GenJetTau3"), branch_names_dict["GenJetInfo"].index("GenJetTau4"))
    else:
        subset = (branch_names_dict["JetInfo"].index("jetMassSoftDrop"), branch_names_dict["JetInfo"].index("jetTau1"), branch_names_dict["JetInfo"].index("jetTau2"), branch_names_dict["JetInfo"].index("jetTau3"), branch_names_dict["JetInfo"].index("jetTau4"))
    subset = sorted(set(subset))
    data_train = data_train[:,subset]
    data_val   = data_val[:,subset]
    data_test  = data_test[:,subset]







predictions = model.predict(data_test)

plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name=modelpath+"Roc")

t_ = 1
bkg = 1
fpr, tpr, thr = roc_curve(labels_test[:,bkg], predictions[:,bkg])
index_01   = (np.abs(fpr - 0.1)).argmin()
index_001  = (np.abs(fpr - 0.01)).argmin()
index_0001 = (np.abs(fpr - 0.001)).argmin()
plt.cla()
plt.xticks( np.arange(0.1,1.1,0.1) )
plt.grid(True, which='both')
plt.axvline(x=tpr[index_01])
plt.axvline(x=tpr[index_001])
plt.axvline(x=tpr[index_0001])
plt.semilogy(tpr, fpr)
# plt.show()

c_ = ["r", "g", "c", "b", "b"]
for bkg, bkg_name in enumerate(sample_names):
    plt.cla()
    fpr, tpr, thr = roc_curve(labels_test[:,bkg], predictions[:,bkg])
    print bkg, str(round(auc(fpr,tpr),3))
    index_01   = (np.abs(fpr - 0.1)).argmin()
    index_001  = (np.abs(fpr - 0.01)).argmin()
    index_0001 = (np.abs(fpr - 0.001)).argmin()
    for bkg1, bkg1_name in enumerate(sample_names):
        plt.hist(predictions[labels_test[:,bkg]==t_][:,bkg1], alpha = 0.5, label=str(t_)+bkg_name+bkg1_name, bins=100)
        plt.axvline(x=thr[index_01], color=c_[bkg])
        plt.axvline(x=thr[index_001], color=c_[bkg])
        plt.axvline(x=thr[index_0001], color=c_[bkg])
    plt.legend(loc='best', shadow=True)
    plt.show()

[branch[i] for i in subset]
