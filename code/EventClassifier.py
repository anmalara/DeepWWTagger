import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy
import json

# from keras.models import Sequential, model_from_json, load_model
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.utils import to_categorical, plot_model
# from keras.optimizers import Adam
# from keras import metrics, regularizers
#
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from RootToKerasFormat import EventTagger
# from UtilsForTraining import *
from variables import *
# from export_model import export_model



class EventClassifier(EventTagger):
    @timeit
    def __init__(self, InputPath = ""):
        EventTagger.__init__(self)
        self.InputPath = InputPath
        self.modelpath = "./"
        self.nObjects = self.TaggerBaseDict["MC_HZ"].nObjects
        self.SubSets = {"Event": tuple([self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Event"].index(var) for var in [ "genInfo.m_weights", "weight_GLP", "weight_lumi"]]),
                        "Jet": tuple([self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Jet"].index(var) for var in self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Jet"]]),
                        "TopJet": tuple([self.TaggerBaseDict["MC_HZ"].VarNameListsDict["TopJet"].index(var) for var in self.TaggerBaseDict["MC_HZ"].VarNameListsDict["TopJet"]]),
                        "Electron": tuple([self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Electron"].index(var) for var in self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Electron"]]),
                        "Muon": tuple([self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Muon"].index(var) for var in self.TaggerBaseDict["MC_HZ"].VarNameListsDict["Muon"]]),
                        }
    @timeit
    def InputShape(self):
        self.LoadVars(self.InputPath)
        data = []
        labels = []
        weights = []
        for Sample in self.SamplesNames:
            sample = []
            for Objects in self.TaggerBaseDict[Sample].ObjectCollection:
                if Objects=="Jet":
                    continue
                nObjects = self.nObjects if Objects!="Event" else 1
                sample.append(self.TaggerBaseDict[Sample].Vars[Objects][:,self.SubSets[Objects]*nObjects])
            sample = np.concatenate(sample,axis=1)
            label = np.ones(sample.shape[0],dtype=int)*self.SamplesNames[Sample]
            weight = np.ones(sample.shape[0],dtype=int)
            data.append(sample)
            labels.append(label)
            weights.append(weight)
        data = np.concatenate(data)
        labels = np.concatenate(labels)
        weights = np.concatenate(weights)
        # labels = to_categorical(labels,num_classes=len(self.SamplesNames))
        self.data_train, self.data_val, self.labels_train, self.labels_val, self.weights_train, self.weights_val, = train_test_split(data, labels, weights, random_state=42, train_size=0.67)
        self.data_val, self.data_test, self.labels_val, self.labels_test, self.weights_val, self.weights_test = train_test_split(self.data_val, self.labels_val, self.weights_val, random_state=42, train_size=0.5)
    @timeit
    def Normalization(data):
        lines = []
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
        return = scaler.transform(data)
        lines.append("# nameBranch NameScaler scaler.mean_ scaler.scale_\n")
        for i in range(len(data)):
            lines.append(i+" "+"MinMaxScaler"+" "+str(scaler.min_[i])+" "+str(scaler.scale_[i])+"\n")
        with open(self.modelpath+"NormInfo.txt", "w") as f:
            for line in lines:
                f.write(line)
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
        # plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, rankdir="LR")
        self.model = model
    @timeit
    def FitModel(self):
        self.model.fit(self.data_train, self.labels_train, sample_weight=self.weights_train, batch_size=self.params["batch_size"], epochs=self.params["epochs"], verbose=1, validation_data=(self.data_val, self.labels_val), callbacks=self.callbacks)
    @timeit
    def Predict(self):
        self.predictions_train = self.model.predict(self.data_train)
        self.predictions_val = self.model.predict(self.data_val)
        self.predictions_test = self.model.predict(self.data_test)
    @timeit
    def Plots(self, show_figure = True, save_figure = False, extraName=""):
        plot_ROC_Curves(self.labels_val, self.predictions_val, self.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"Roc"+extraName)
        plot_ROC_Curves1vs1(self.labels_val, self.predictions_val, self.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"Roc1vs1"+extraName)
        plot_outputs_1d(self, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = self.modelpath+"Outputs"+extraName)
        NNResponce(self.labels_val, self.predictions_val, self.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = self.modelpath+"NNResponce"+extraName)
        MaximiseSensitivity(self.labels_val, self.predictions_val, self.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = self.modelpath+"Sensistivity"+extraName)
        if self.isNew:
            plot_losses(self.callbacks[0], min_epoch=0,  losses="loss", show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"loss"+extraName)
            plot_losses(self.callbacks[0], min_epoch=0,  losses="acc",  show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"acc"+extraName)
            plot_losses(self.callbacks[0], min_epoch=10, losses="loss", show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"loss_10"+extraName)
            plot_losses(self.callbacks[0], min_epoch=10, losses="acc",  show_figure=show_figure, save_figure=save_figure, name=self.modelpath+"acc_10"+extraName)
    @timeit
    def SaveModel(self):
        with open(self.modelpath+"mymodeljson.json", "w") as f:
            f.write(str(self.model.to_json()) + "\n")
        self.model.save_weights(self.modelpath+"myweights.h5")
        self.model.save(self.modelpath+"mymodel.h5")
        export_model(self.model, self.modelpath+"mymodel.txt")



EC = EventClassifier(filePath)
EC.InputShape()
EC.Normalization()





filePath = "/beegfs/desy/user/amalara/input_varariables/"
# filePath = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"
# filePath = "/Users/andrea/Desktop/Analysis/"


dict_var = {"Info_dict": Info_dict,
            "isSubset" : True,
            "isGen" : False,
            "filePath" : "/beegfs/desy/user/amalara/",
            "modelpath" : "model_newarch",
            "varnames" : ["JetInfo", "JetVariables"],
            # "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "ncandidates", "jetBtag", "jetTau1", "jetTau2", "jetTau3", "jetTau21", "jetTau31", "jetTau32"],
            "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetTau1", "jetTau2"],
            "sample_names": sorted(["Higgs", "QCD", "Top" ]),
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : -1,
            "layers" : [50,200,200,50,10],
            "params" : {"epochs" : 10, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

NN = EventClassifier(dict_var)
# NN.InputShape()
# NN.CreateSubSet()
# NN.Normalization()
