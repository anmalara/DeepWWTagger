import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy
import json

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from UtilsForTraining import *
from variables import *
from export_model import export_model

class SequentialNN:
    @timeit
    def __init__(self, dict_var):
        self.sample_names = dict_var["sample_names"]
        self.Info_dict = dict_var["Info_dict"]
        self.varnames = dict_var["varnames"]
        self.variables = dict_var["variables"]
        self.branches = [var for vars in self.varnames for var in self.Info_dict[vars] ]
        self.radius = dict_var["radius"]
        self.pt_min = str(dict_var["pt_min"])
        self.pt_max = str(dict_var["pt_max"])
        self.isSubset = dict_var["isSubset"]
        self.isGen = dict_var["isGen"]
        self.max_size = dict_var["max_size"]
        self.filePath = dict_var["filePath"]
        self.inputFolder = self.filePath+"input_varariables/NTuples_Tagger/Inputs/"
        self.modelName = "model_"+self.radius+"_pt_"+self.pt_min+"_"+self.pt_max+"/"
        self.outputFolder = self.filePath+"output_varariables/Sequential/"+self.modelName
        self.modelpath = self.outputFolder+dict_var["modelpath"]+"/"
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)
        else:
            for file in glob(self.modelpath+"*"):
                os.remove(file)
        self.layers = dict_var["layers"]
        self.params = dict_var["params"]
        with open(self.modelpath+"mymodelinfo.json","w") as f:
            f.write(json.dumps(dict_var))
        with open(self.modelpath+"mymodelinfo.txt","w") as f:
            f.write(str(dict_var))
    @timeit
    def InputShape(self):
        @timeit
        def loadVariables(sample_name):
            subcounter = 0
            sample = []
            for file_ in glob(self.inputFolder+sample_name+"_"+self.radius+"/"+self.varnames[0]+"/file_*_pt_"+self.pt_min+"_"+self.pt_max+".npy"):
                temp = np.load(file_, encoding = "bytes")
                for i in range(1,len(self.varnames)):
                    temp = np.hstack((temp, np.load(file_.replace(self.varnames[0], self.varnames[i]), encoding = "bytes")))
                temp = temp[~np.isinf(temp).any(axis=(1))]
                temp = temp[~np.isnan(temp).any(axis=(1))]
                sample.append(temp)
                subcounter += temp.shape[0]
                if subcounter > self.max_size and self.max_size != -1:
                    break
            return sample
        data = []
        labels = []
        for i, sample_name in enumerate(self.sample_names):
            sample = loadVariables(sample_name)
            sample = np.concatenate(sample)
            print(sample_name, sample.shape)
            sample = sample[:self.max_size,:]
            print(sample_name, sample.shape)
            label = np.ones(sample.shape[0],dtype=int)*i
            data.append(sample)
            labels.append(label)
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        labels = to_categorical(labels,num_classes=len(self.sample_names))
        self.data_train, self.data_val, self.labels_train, self.labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
        self.data_val, self.data_test, self.labels_val, self.labels_test = train_test_split(self.data_val, self.labels_val, random_state=42, train_size=0.5)
    @timeit
    def CreateSubSet(self):
        self.subset = tuple(sorted(set([self.branches.index(var) for var in self.variables])))
        self.data_train = self.data_train[:,self.subset]
        self.data_val   = self.data_val[:,self.subset]
        self.data_test  = self.data_test[:,self.subset]
    @timeit
    def Normalization(self):
        def FindScaler(name):
            if name=="ncandidates": return "StandardScalerNoMean"
            elif name=="jetEta": return "StandardScaler"
            else: return "MinMaxScaler"
        with open(self.modelpath+"NormInfo.txt", "w") as f:
            f.write("# nameBranch NameScaler scaler.mean_[0] scaler.scale_\n")
            for iBranch in self.subset:
                nameBranch = self.branches[iBranch]
                indexBranch = (self.variables.index(nameBranch),)
                NameScaler = FindScaler(nameBranch)
                col_standard = self.data_train[:,indexBranch]
                if nameBranch == "jetBtag":
                    continue
                if nameBranch == "jetMassSoftDrop" or nameBranch == "jetBtag":
                    col_standard = col_standard[col_standard[:,0]>0,:]
                if NameScaler == "StandardScaler":
                    scaler = preprocessing.StandardScaler().fit(col_standard)
                    f.write(nameBranch+" "+NameScaler+" "+str(scaler.mean_[0])+" "+str(scaler.scale_[0])+"\n")
                if NameScaler == "MinMaxScaler":
                    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(col_standard)
                    f.write(nameBranch+" "+NameScaler+" "+str(scaler.min_[0])+" "+str(scaler.scale_[0])+"\n")
                if NameScaler == "StandardScalerNoMean":
                    scaler = preprocessing.StandardScaler(with_mean=False).fit(col_standard)
                    f.write(nameBranch+" "+NameScaler+" "+"[0]"+" "+str(scaler.scale_[0])+"\n")
                for X in [self.data_train,self.data_val,self.data_test]:
                    X[:,indexBranch] = scaler.transform(X[:,indexBranch])
    @timeit
    def SequentialModel(self):
        model = Sequential()
        # Define layers
        input_shape = self.data_train.shape[1]
        output_shape = self.labels_train.shape[1]
        model.add(Dense(self.layers[0], input_dim=input_shape, activation=self.params["activation"],kernel_initializer=self.params["kernel_initializer"]))
        for i in range(1,len(self.layers)):
            model.add(Dense(self.layers[i], activation=self.params["activation"],kernel_initializer=self.params["kernel_initializer"],bias_initializer=self.params["bias_initializer"]))
            model.add(BatchNormalization())
            model.add(Dropout(self.params["dropoutRate"]))
        model.add(Dense(output_shape, activation=self.params["activation_last"]))
        # Compile
        self.myloss = "categorical_crossentropy"
        if output_shape == 1:
            self.myloss = "binary_crossentropy"
        model.compile(loss=self.myloss, optimizer=self.params["optimizer"], metrics=self.params["metrics"])
        model.summary()
        self.callbacks = DefineCallbacks(self.modelpath)
        # TODO
        # save model params
        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
        self.model = model
    @timeit
    def FitModel(self):
        self.model.fit(self.data_train, self.labels_train, batch_size=self.params["batch_size"], epochs=self.params["epochs"], verbose=1, validation_data=(self.data_val, self.labels_val), callbacks=self.callbacks)
    @timeit
    def Predict(self):
        self.predictions = self.model.predict(self.data_test)
    @timeit
    def Plots(self, show_figure = True, save_figure = False):
        PlotInfos(self.labels_test, self.predictions, self.sample_names, self.callbacks[0], self.modelpath, show_figure = show_figure, save_figure = save_figure)
    @timeit
    def SaveModel(self):
        with open(self.modelpath+"mymodeljson.json", 'w') as f:
            f.write(str(self.model.to_json()) + '\n')
        self.model.save_weights(self.modelpath+"myweights.h5")
        self.model.save(self.modelpath+"mymodel.h5")
        export_model(self.model, self.modelpath+"mymodel.txt")

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

filePath = "/beegfs/desy/user/amalara/"
# filePath = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"
# filePath = "/Users/andrea/Desktop/Analysis/"


dict_var = {"Info_dict": Info_dict,
            "isSubset" : True,
            "isGen" : False,
            "filePath" : "/beegfs/desy/user/amalara/",
            "modelpath" : "model",
            "varnames" : ["JetInfo", "JetVariables"],
            "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetTau1", "jetTau2", "jetTau3", "ncandidates", "jetBtag", "jetTau21", "jetTau31", "jetTau32"],
            "sample_names": sorted(["Higgs", "QCD", "Top" ]),
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : 10000,
            "layers" : [50,50,50,50,50,50,10],
            "params" : {"epochs" : 100, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

NN = SequentialNN(dict_var)
NN.InputShape()
NN.CreateSubSet()
NN.Normalization()
NN.SequentialModel()
NN.FitModel()
NN.Predict()
NN.Plots(show_figure = False, save_figure = True)
NN.SaveModel()

quit()

#
#
#
#layers = {"input": [data_train.shape[1],50], "layers": [50,50,50,50,50,50,50,10], "output": [labels_train.shape[1]]}
#
#i=0
#
#par_list = []
#auc_list = []
#
#for batch_size in [10, 128, 1024, 4096]:
#    for epochs in [2]:
#        for optimizer in ['SGD','Adam',"adam"]:
#            for dropoutRate in [0.001, 0.01, 0.2]:
#                for kernel_initializer in ['lecun_uniform','zero', 'glorot_normal']:
#                    for bias_initializer in ['lecun_uniform','zero', 'glorot_normal']:
#                        for activation in ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']:
#                            i += 0
#                            print i
#                            params = {"activation" : activation, "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer, "activation_last": "softmax", "optimizer": optimizer, "metrics":["accuracy"], "dropoutRate": dropoutRate}
#                            params["batch_size"] = batch_size
#                            params["epochs"] = epochs
#                            seed = 4444
#                            np.random.seed(seed)
#                            model = SequentialModel(layers, params)
#                            callbacks = DefineCallbacks(modelpath)
#                            model.fit(data_train, labels_train, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1, validation_data=(data_val,labels_val), callbacks=callbacks)
#                            predictions = model.predict(data_test)
#                            fpr, tpr, thr = roc_curve(labels_test[:,sample_names.index("Higgs")], predictions[:,sample_names.index("Higgs")])
#                            print params
#                            par_list.append(params)
#                            auc_list.append(auc(fpr,tpr))
#                            print "Auc:", auc(fpr,tpr)
#
#quit()
#
#

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])




#
#
# dict_var = {"Info_dict": Info_dict,
#             "isSubset" : True,
#             "isGen" : False,
#             "filePath" : "/beegfs/desy/user/amalara/",
#             "modelpath" : "model_analyze",
#             "varnames" : ["JetInfo", "JetVariables"],
#             "variables" : ["jetPt", "jetEta"],
#             "sample_names": sorted(["Higgs", "QCD", "Top" ]),
#             "radius" : "AK8",
#             "pt_min" : 300,
#             "pt_max" : 500,
#             "max_size" : 10000,
#             "layers" : [3,4,2],
#             "params" : {"epochs" : 12, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
#             "seed" : 4444
#             }
#
#
#
# dict_var = {"Info_dict": Info_dict,
#             "isSubset" : True,
#             "isGen" : False,
#             "filePath" : "/beegfs/desy/user/amalara/",
#             "modelpath" : "model_analyze",
#             "varnames" : ["JetInfo", "JetVariables"],
#             "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetTau1", "jetTau2", "jetTau3", "ncandidates", "jetBtag", "jetTau21", "jetTau31", "jetTau32"],
#             "sample_names": sorted(["Higgs", "QCD", "Top" ]),
#             "radius" : "AK8",
#             "pt_min" : 300,
#             "pt_max" : 500,
#             "max_size" : 10000,
#             "layers" : [50,50,50,50,50,50,10],
#             "params" : {"epochs" : 30, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
#             "seed" : 4444
#             }
#
# np.random.seed(dict_var["seed"])
#
# NN = SequentialNN(dict_var)
# NN.InputShape()
# NN.CreateSubSet()
# NN.Normalization()
# NN.SequentialModel()
# NN.FitModel()
# NN.Predict()
# NN.SaveModel()
#
# def RELU(v):
#     return np.maximum(v, 0)
#
#
# def softmax(v):
#     x = v[0,:]
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
# model = NN.model
# arch = json.loads(open(NN.modelpath+"mymodeljson.json").read())
#
# i_i = 10
# sum = 0
#
# for i_i in range(0,max_):
#     inputs = NN.data_train[i_i:i_i+1,:]
#     outputs = NN.model.predict(NN.data_train[i_i:i_i+1,:])
#     test = NN.data_train[i_i:i_i+1,:]
#     for ind, l in enumerate(model.get_config()):
#         # print l['class_name']
#         if l['class_name'] == "Dropout":
#             continue
#         # print l['class_name']
#         if l['class_name'] == "Dense":
#             # print len(model.layers[ind].get_weights()), model.layers[ind].get_weights()[0].shape, model.layers[ind].get_weights()[1].shape
#             test = np.matmul(test, model.layers[ind].get_weights()[0])
#             test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#             if l["config"]["activation"] == "relu":
#                 test = RELU(test)
#             if l["config"]["activation"] == "softmax":
#                 test = softmax(test)
#         if l['class_name'] == "BatchNormalization":
#             test = test - np.expand_dims(model.layers[ind].get_weights()[2],axis=0)
#             test = test / np.sqrt((np.expand_dims(model.layers[ind].get_weights()[3],axis=0)+0.001))
#             test = test * np.expand_dims(model.layers[ind].get_weights()[0],axis=0)
#             test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#     sum = np.abs(test- outputs) + sum
#
# print sum
#
#
# max_ = 100
# for value in [0,1,2,3]:
#     add = [0,0,0,0]
#     for i_i in range(0,max_):
#         min = 10
#         index = 0
#         inputs = NN.data_train[i_i:i_i+1,:]
#         outputs = NN.model.predict(NN.data_train[i_i:i_i+1,:])
#         test_ = 0
#         for p in multiset_permutations(np.array([0, 1, 2, 3])):
#             test = NN.data_train[i_i:i_i+1,:]
#             for ind, l in enumerate(arch["config"]):
#                 # print l['class_name']
#                 if l['class_name'] == "Dropout":
#                     continue
#                 # print l['class_name']
#                 if l['class_name'] == "Dense":
#                     # print len(model.layers[ind].get_weights()), model.layers[ind].get_weights()[0].shape, model.layers[ind].get_weights()[1].shape
#                     test = np.matmul(test, model.layers[ind].get_weights()[0])
#                     test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#                     if l["config"]["activation"] == "relu":
#                         test = RELU(test)
#                     if l["config"]["activation"] == "softmax":
#                         test = softmax(test)
#                 if l['class_name'] == "BatchNormalization":
#                     test = test - np.expand_dims(model.layers[ind].get_weights()[p[0]],axis=0)
#                     test = test / np.sqrt((np.expand_dims(model.layers[ind].get_weights()[p[1]],axis=0)+0.001))
#                     test = test * np.expand_dims(model.layers[ind].get_weights()[p[2]],axis=0)
#                     test = test + np.expand_dims(model.layers[ind].get_weights()[p[3]],axis=0)
#             temp = np.sum(np.abs(test-outputs))
#             if temp < min :
#                 min = temp
#                 index = p
#                 test_ = test
#         print min, index
#         print test_
#         print outputs
#         for t in [0,1,2,3]:
#             if index[t] == value :
#                 add[t] = add[t] + 1./max_
#     print value, add
