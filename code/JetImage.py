import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy
import json

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from UtilsForTraining import *
from variables import *
from exportmodel import export_model

class JetImageNN:
    @timeit
    def __init__(self, dict_var):
        self.sample_names = dict_var["sample_names"]
        self.Info_dict = dict_var["Info_dict"]
        self.varnames = dict_var["varnames"]
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
        self.outputFolder = self.filePath+"output_varariables/JetImage/"+self.modelName
        self.modelpath = self.outputFolder+dict_var["modelpath"]+"/"
        if not os.path.exists(self.modelpath):
            os.makedirs(self.modelpath)
        else:
            for file in glob(self.modelpath+"*"):
                os.remove(file)
        self.layers = dict_var["layers"]
        self.params = dict_var["params"]
        self.Dense = dict_var["Dense"]
        self.kernel_size = dict_var["kernel_size"]
        self.strides = dict_var["strides"]
        self.params = dict_var["params"]
        with open(self.modelpath+"mymodelinfo.json","w") as f:
            f.write(json.dumps(dict_var))
        with open(self.modelpath+"mymodelinfo.txt","w") as f:
            f.write(str(dict_var))
    @timeit
    def InputShape(self):
        data = []
        labels = []
        for i, sample_name in enumerate(self.sample_names):
            subcounter = 0
            sample = []
            for file_ in glob(self.inputFolder+sample_name+"_"+self.radius+"/"+self.varnames[0]+"/file_*_pt_"+self.pt_min+"_"+self.pt_max+".npy"):
                temp = np.load(file_, encoding = "bytes")
                if len(temp.shape)<4:
                    continue
                temp = temp[~np.isinf(temp).any(axis=(1,2,3))]
                sample.append(temp)
                subcounter += temp.shape[0]
                if subcounter > self.max_size and self.max_size != -1:
                    break
            sample = np.concatenate(sample)
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
    def JetImageModel(self):
        model = Sequential()
        # Define layers
        input_shape = self.data_train.shape[1:4]
        output_shape = self.labels_train.shape[1]
        model.add(Conv2D(self.layers[0], input_shape=input_shape, kernel_size=self.kernel_size[0], strides=self.strides[0], padding=self.params["padding"], activation=self.params["activation"], data_format=self.params["data_format"]))
        for i in range(1,len(self.layers)):
            model.add(Conv2D(self.layers[i], kernel_size=self.kernel_size[i], strides=self.strides[i], padding=self.params["padding"], activation=self.params["activation"], data_format=self.params["data_format"]))
            # model.add(BatchNormalization())
            model.add(Dropout(self.params["dropoutRate"]))
        model.add(Flatten())
        for i in range(0,len(self.Dense)):
            model.add(Dense(self.Dense[i], activation=self.params["activation"],kernel_initializer=self.params["kernel_initializer"],bias_initializer=self.params["bias_initializer"]))
            # model.add(BatchNormalization())
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
            "varnames" : ["JetImage"],
            "sample_names": sorted(["QCD", "Top" ]),
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : 1000000,
            "layers" : [48,24,12,2],
            "Dense" : [40,30,10],
            "kernel_size" : [(2,2),(3,3),(3,3),(3,3)],
            "strides" : [(1,1),(2,2),(2,2),(2,2)],
            "params" : {"epochs" : 10, "batch_size" : 128, "padding" : "valid", "data_format" : "channels_first", "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

NN = JetImageNN(dict_var)
NN.InputShape()
NN.JetImageModel()
NN.FitModel()
NN.Predict()
NN.Plots(show_figure = True, save_figure = True)
NN.Plots(show_figure = False, save_figure = True)
NN.SaveModel()

quit()

json_string = model.to_json()
with open(modelpath+"mymodeljson.txt", 'w') as f:
    f.write(str(model.to_json()) + '\n')

model.save_weights(modelpath+"myweights.h5")
model.save(modelpath+"mymodel.h5")

# PlotInfos(labels_test, predictions, sample_names,callbacks[0],modelpath)
PlotInfos(labels_test, predictions, sample_names,callbacks[0],modelpath, show_figure = True, save_figure = False)

quit()

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])
