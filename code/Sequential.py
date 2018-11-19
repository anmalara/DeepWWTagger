import numpy as np
from math import *
import os.path
import os
import sys
import copy

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from variables import *

seed = 0
np.random.seed(seed)

def SequentialModel(layers = {"input": [5,20], "layers": [10,10,10], "output": [1]}, params = {"activation" : "relu", "kernel_initializer": "lecun_uniform", "bias_initializer": "zeros", "activation_last": "sigmoid"}):
    model = Sequential()
    # Define layers
    model.add(Dense(layers["input"][1], input_dim=layers["input"][0], activation=params["activation"],kernel_initializer=params["kernel_initializer"]))
    for i in range(0,len(layers["layers"])):
        model.add(Dense(layers["layers"][i], activation=params["activation"],kernel_initializer=params["kernel_initializer"],bias_initializer=params["bias_initializer"]))
        model.add(BatchNormalization())
        model.add(Dropout(params["dropoutRate"]))
    model.add(Dense(layers["output"][0], activation=params["activation_last"]))
    # Compile
    if layers["output"][0]==1:
        myloss = "binary_crossentropy"
    else:
        myloss = "categorical_crossentropy"
    model.compile(loss=myloss, optimizer=params["optimizer"], metrics=params["metrics"])
    model.summary()
    # TODO
    # save model params
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
    return model



def plot_ROC_Curve(tpr, fpr, namePlot, title, outputName ):
    plt.cla()
    plt.figure()
    plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
    plt.grid(True, which='both')
    plt.semilogy(tpr, fpr, label=namePlot)
    # for i in range(0,len(tpr)):
    #     plt.semilogy(tpr[i], fpr[i], label=namePlot[i])
    plt.semilogy([0.001, 1], [0.001, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.001, 1.05])
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background mistag rate")
    plt.title(title)
    plt.legend(loc='best', shadow=True)
    plt.title('ROC curve (area = %0.2f)' % roc_auc)
    plt.savefig(outputName)


def loadVariables(bkg):
    temp = inputFolder+"Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
    return np.load(temp, encoding = "bytes")

def InputShape(sample_names=["Higgs","QCD", "Top"],max_size=500000):
    First = True
    for i, sample_name in enumerate(sample_names):
        sample = loadVariables(sample_name)
        print(sample_name, sample.shape)
        sample = sample[:max_size,:]
        print(sample_name, sample.shape)
        label = np.ones(sample.shape[0],dtype=int)*i
        if First:
            data = sample
            labels = label
            First = False
        else:
            data = np.concatenate((data, sample))
            labels = np.concatenate((labels, label))
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, random_state=42, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test


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
        else: return "MaxAbsScaler"
    for nameBranch in branch_names_dict[info]:
        if nameBranch=="jetBtag":
            continue
        indexBranch = branch_names_dict[info].index(nameBranch)
        col_standard = X_standard[:,(indexBranch,)].astype(variable_type)
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


def IndexingMatrix(matrix, matrix_check, check):
    return matrix[np.asarray([np.array_equal(el,check) for el in matrix_check])]


def plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True):
    classes = to_categorical(np.arange(len(sample_names)))
    plt.cla()
    plt.xticks( np.arange(0.1,1.1,0.1) )
    plt.grid(True, which='both')
    for i, sample in enumerate(sample_names):
        fpr, tpr, thr = roc_curve(labels_test[:,i], predictions[:,i])
        mean = IndexingMatrix(predictions, labels_test, classes[i]).mean(axis=0)
        std = IndexingMatrix(predictions, labels_test, classes[i]).std(axis=0)
        label = sample+": auc = "+str(round(auc(fpr,tpr),3))+", mean = ["
        for j in range(len(sample_names)):
            label += str(round(mean[j],3))+"+-"+str(round(std[j],3))+","
        label += "]"
        if isLogy:
            plt.semilogy(tpr, fpr, label=label)
        else:
            plt.plot(tpr, fpr, label=label)
    x= np.linspace(0.001, 1,1000)
    if isLogy:
        plt.semilogy(x, x, label = "Random classifier: auc = 0.5 ")
    else:
        plt.plot(x, x, label = "Random classifier: auc = 0.5 ")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.001, 1.05])
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background mistag rate")
    plt.legend(loc='best', shadow=True)
    plt.show()


def plot_losses(hist, show_figure=True, save_figure=False, losses="loss", min_epoch=0):
    plt.clf()
    plt.plot(hist.history[losses][min_epoch:], label="Training "+losses)
    plt.plot(hist.history["val_"+losses][min_epoch:], label="Validation "+losses)
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save_figure:
        if isinstance(save_figure, bool):
            save_figure = "history.png"
        plt.savefig(save_figure)
    if show_figure:
        plt.show()



def ResizeInput(datas, subset):
    if not isinstance(subset,tuple):
        subset = tuple(np.arange(0,datas[0].shape[1]))
    else:
        subset = sorted(set(subset))
        if subset[-1]>datas[0].shape[1]:
            subset = tuple(np.arange(0,datas[0].shape[1]))
    print(subset)
    for index, data in enumerate(datas):
        datas[index] = data[:,subset]
    for data in datas:
        print(data.shape)


##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

try:
    file_min = int(sys.argv[1])
    file_max = int(sys.argv[2])
    pt_min = int(sys.argv[3])
    pt_max = int(sys.argv[4])
    info = sys.argv[5]
    radius = sys.argv[6]
    isSubset = int(sys.argv[7])
    sample_names = ["Higgs", "QCD", "Top"]
except:
    file_min = 0
    file_max = 950
    pt_min = 300
    pt_max = 500
    info = "JetInfo"
    radius = "AK8"
    isSubset = False
    sample_names = ["Higgs", "QCD", "Top"]

file_min = 0
file_max = 1000
pt_min = 300
pt_max = 500
info = "JetInfo"
radius = "AK15"
isSubset = True
sample_names = ["Higgs", "QCD", "Top"]

modelName = "model_"+info+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+"/"
filePath = "/beegfs/desy/user/amalara/"
# filePath = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"
# filePath = "/Users/andrea/Desktop/Analysis/"

inputFolder = filePath+"input_varariables/NTuples_Tagger/Sequential/"
outputFolder = filePath+"output_varariables/Sequential/"+modelName
modelpath = outputFolder+"models/"

if not os.path.exists(modelpath):
    os.makedirs(modelpath)

isGen = 0
subset = None
if "Gen" in info:
    isGen = 1

if isSubset:
    if isGen:
        subset = (branch_names_dict["GenJetInfo"].index("GenSoftDropMass"), branch_names_dict["GenJetInfo"].index("GenJetTau1"), branch_names_dict["GenJetInfo"].index("GenJetTau2"), branch_names_dict["GenJetInfo"].index("GenJetTau3"), branch_names_dict["GenJetInfo"].index("GenJetTau4"))
    else:
        subset = (branch_names_dict["JetInfo"].index("jetMassSoftDrop"), branch_names_dict["JetInfo"].index("jetTau1"), branch_names_dict["JetInfo"].index("jetTau2"), branch_names_dict["JetInfo"].index("jetTau3"), branch_names_dict["JetInfo"].index("jetTau4"))
    subset = sorted(set(subset))


data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names,max_size=-1)
data_train.shape

data_train_ = copy.deepcopy(data_train)
data_val_ = copy.deepcopy(data_val)
data_test_ = copy.deepcopy(data_test)
labels_train_ = copy.deepcopy(labels_train)
labels_val_ = copy.deepcopy(labels_val)
labels_test_ = copy.deepcopy(labels_test)


max = len(data_train_)
data_train = data_train_[:max,:]
data_val = data_val_[:max,:]
data_test = data_test_[:max,:]
labels_train = labels_train_[:max,:]
labels_val = labels_val_[:max,:]
labels_test = labels_test_[:max,:]

data_train.shape
data_train = data_train.astype(variable_type)
data_val = data_val.astype(variable_type)
data_test = data_test.astype(variable_type)

data_train.mean(axis=0)
data_test.mean(axis=0)
data_val.mean(axis=0)

data_train.shape
Normalization(data_train,[data_train,data_val,data_test])

data_train.shape
data_train.mean(axis=0)
data_test.mean(axis=0)
data_val.mean(axis=0)


data_train = data_train[:,subset]
data_val = data_val[:,subset]
data_test = data_test[:,subset]


data_train.shape

layers = {"input": [data_train.shape[1],50], "layers": [50,50,50,50,50,50,50,10], "output": [labels_train.shape[1]]}
# params = {"activation" : "relu", "kernel_initializer": "lecun_uniform", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"]}
params = {"activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01}
params["batch_size"] = 128
params["epochs"] = 10

seed = 4444
np.random.seed(seed)
model = SequentialModel(layers, params)

callbacks = []
history = History()
callbacks.append(history)
modelCheckpoint_loss      = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor='val_loss', save_best_only=False)
callbacks.append(modelCheckpoint_loss)
modelCheckpoint_acc       = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor='val_acc', save_best_only=False)
callbacks.append(modelCheckpoint_acc)
modelCheckpoint_loss_best = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True)
callbacks.append(modelCheckpoint_loss_best)
modelCheckpoint_acc_best  = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor='val_acc', save_best_only=True)
callbacks.append(modelCheckpoint_acc_best)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_delta=0.01, min_lr=0.001, cooldown=10)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.001, cooldown=10)
# callbacks.append(reduceLROnPlateau)

model.fit(data_train, labels_train, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1, validation_data=(data_val,labels_val), callbacks=callbacks)

predictions = model.predict(data_test)


plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True)


plot_losses(history, min_epoch=0, losses="loss")
plot_losses(history, min_epoch=10, losses="loss")
plot_losses(history, min_epoch=0, losses="acc")
plot_losses(history, min_epoch=10, losses="acc")


quit()



with open(modelpath+"config.txt", 'w') as f:
    for s in config_:
        f.write(str(s) + '\n')


with open(modelpath+"modeljson.txt", 'w') as f:
    f.write(str(json_string) + '\n')

with open(modelpath+"modelyaml.txt", 'w') as f:
    f.write(str(yaml_string) + '\n')


#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])
