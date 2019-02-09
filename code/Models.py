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
    def __init__(self, dict_var, isNew=True):
        self.sample_names = dict_var["sample_names"]
        self.isNew = isNew
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
        self.layers = dict_var["layers"]
        self.params = dict_var["params"]
        if self.isNew:
            if not os.path.exists(self.modelpath):
                os.makedirs(self.modelpath)
            else:
                for file in glob(self.modelpath+"*"):
                    os.remove(file)
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
        self.subset = tuple([self.branches.index(var) for var in self.variables])
        self.data_train = self.data_train[:,self.subset]
        self.data_val   = self.data_val[:,self.subset]
        self.data_test  = self.data_test[:,self.subset]
    @timeit
    def Normalization(self):
        def FindScaler(name):
            if name=="ncandidates": return "StandardScalerNoMean"
            elif name=="jetEta": return "StandardScaler"
            else: return "MinMaxScaler"
        lines = []
        lines.append("# nameBranch NameScaler scaler.mean_[0] scaler.scale_\n")
        for iBranch in self.subset:
            nameBranch = self.branches[iBranch]
            indexBranch = (self.variables.index(nameBranch),)
            NameScaler = FindScaler(nameBranch)
            col_standard = self.data_train[:,indexBranch]
            if nameBranch == "jetMassSoftDrop" or nameBranch == "jetBtag":
                col_standard = col_standard[col_standard[:,0]>0,:]
            if NameScaler == "StandardScaler":
                scaler = preprocessing.StandardScaler().fit(col_standard)
                lines.append(nameBranch+" "+NameScaler+" "+str(scaler.mean_[0])+" "+str(scaler.scale_[0])+"\n")
            if NameScaler == "MinMaxScaler":
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(col_standard)
                lines.append(nameBranch+" "+NameScaler+" "+str(scaler.min_[0])+" "+str(scaler.scale_[0])+"\n")
            if NameScaler == "StandardScalerNoMean":
                scaler = preprocessing.StandardScaler(with_mean=False).fit(col_standard)
                print scaler.mean_[0]
                lines.append(nameBranch+" "+NameScaler+" "+"0"+" "+str(scaler.scale_[0])+"\n")
            for X in [self.data_train,self.data_val,self.data_test]:
                X[:,indexBranch] = scaler.transform(X[:,indexBranch])
        if self.isNew:
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
        # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
        self.model = model
    @timeit
    def FitModel(self):
        self.model.fit(self.data_train, self.labels_train, batch_size=self.params["batch_size"], epochs=self.params["epochs"], verbose=1, validation_data=(self.data_val, self.labels_val), callbacks=self.callbacks)
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
        with open(self.modelpath+"mymodeljson.json", 'w') as f:
            f.write(str(self.model.to_json()) + '\n')
        self.model.save_weights(self.modelpath+"myweights.h5")
        self.model.save(self.modelpath+"mymodel.h5")
        export_model(self.model, self.modelpath+"mymodel.txt")
