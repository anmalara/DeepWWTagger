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
        with open(self.modelpath+"NormInfo.txt", "w") as f:
            f.write("# nameBranch NameScaler scaler.mean_[0] scaler.scale_\n")
            for iBranch in self.subset:
                nameBranch = self.branches[iBranch]
                indexBranch = (self.variables.index(nameBranch),)
                NameScaler = FindScaler(nameBranch)
                col_standard = self.data_train[:,indexBranch]
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
                    print scaler.mean_[0]
                    f.write(nameBranch+" "+NameScaler+" "+"0"+" "+str(scaler.scale_[0])+"\n")
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
            "modelpath" : "model_test",
            "varnames" : ["JetInfo", "JetVariables"],
            "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "ncandidates", "jetBtag", "jetTau1", "jetTau2", "jetTau3", "jetTau21", "jetTau31", "jetTau32"],
            "sample_names": sorted(["Higgs", "QCD", "Top" ]),
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : 15000,
            "layers" : [50,50,50,50,50,50,10],
            "params" : {"epochs" : 100, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

NN = SequentialNN(dict_var)
NN.InputShape()
NN.CreateSubSet()
pretest = copy.deepcopy(NN.data_train)
test = copy.deepcopy(NN.data_train)
NN.Normalization()
posttest = copy.deepcopy(NN.data_train)
# NN.SequentialModel()
# NN.FitModel()
# NN.Predict()
# NN.Plots(show_figure = False, save_figure = True)
# NN.SaveModel()
#
# quit()


# test = np.array()

modelname = "model_300epochs_1500k"
modelname = "model_test"
# model = load_model("/beegfs/desy/user/amalara/output_varariables/Sequential/model_AK8_pt_300_500/model_300epochs_1500k/mymodel.h5")

variables = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "ncandidates", "jetBtag", "jetTau1", "jetTau2", "jetTau3", "jetTau21", "jetTau31", "jetTau32"]

means = np.zeros(len(variables))
stds  = np.ones(len(variables))
NfileName = "/beegfs/desy/user/amalara/output_varariables/Sequential/model_AK8_pt_300_500/"+modelname+"/NormInfo.txt"

with open(NfileName, "U") as file:
    lines = file.readlines()
    i=0
    for line in lines[1:]:
        means[i] = float(line.split()[2])
        stds[i] = float(line.split()[3])
        i += 1


means[variables.index("jetEta")] = -means[variables.index("jetEta")]/stds[variables.index("jetEta")]
stds[variables.index("jetEta")] = 1./stds[variables.index("jetEta")]

means[variables.index("ncandidates")] = -means[variables.index("ncandidates")]/stds[variables.index("ncandidates")]
stds[variables.index("ncandidates")] = 1./stds[variables.index("ncandidates")]


test = copy.deepcopy(pretest)
test *= stds
test += means

test - posttest


test = np.array([375.60577, 1.5032118, 2.1382484, 5489.4786, 894.05535, 0.2823343, 0.1399507, 0.0921988, 0.6395325, 86, 0.4956915, 0.3265591, 0.6587950])
# test
test = np.array([438.78707, 0.5129144, 3.0206136, 50.267421, 509.01425, 0.4989742, 0.2019137, 0.1303722, 0.0840893, 84, 0.6456830, 0.4164616, 0.6449938])
# test/pretest[0]
# test = copy.deepcopy(pretest[0])
# test
test *= stds
test += means
test = np.expand_dims(test,axis=0)
model.predict(test)



test = []
test = np.array(test)

test *= stds
test += means
model.predict(test)




test = np.array([311.001, -0.0329501, -0.286615, 94.3219, 325.151, 0.109593, 0.272546, 0.0777287, 0.064548, 53, 0.285195, 0.236834, 0.830427])



test = []
out = []

test.append([304.992, -0.770511, 2.09472, 121.131, 418.03, 0.191913, 0.451919, 0.264925, 0.165686, 73, 0.586223, 0.366627, 0.625406])
out.append([3.30935e-11, 0.999923, 7.72649e-05])
test.append([323.679, 0.603732, 0.810878, 116.488, 401.741, 0.930311, 0.349647, 0.223829, 0.130089, 65, 0.640156, 0.372059, 0.581201])
out.append([2.23748e-09, 0.94344, 0.0565604])

test.append([348.36, 0.66382, -3.04273, 144.417, 451.684, 0.785437, 0.461707, 0.203139, 0.165481, 88, 0.439974, 0.358411, 0.81462])
out.append([1.53249e-15, 0.993052, 0.00694815])

test.append([411.751, 0.683111, 1.44729, 126.518, 527.027, 0.517831, 0.327576, 0.1169, 0.0714266, 77, 0.356862, 0.218046, 0.611009])
out.append([3.17041e-11, 0.999919, 8.11899e-05])

test.append([362.901, 2.02711, 0.928488, 158.563, 1410.43, 0.400895, 0.421436, 0.219683, 0.136605, 59, 0.521272, 0.324142, 0.621829])
out.append([4.94501e-10, 0.999987, 1.33914e-05])

test.append([340.085, -0.600769, 1.84248, 108.915, 417.773, 0.756932, 0.320598, 0.149482, 0.0969606, 90, 0.466258, 0.302436, 0.648646])
out.append([1.3181e-15, 0.991037, 0.00896344])

test.append([427.57, -1.28022, -2.67261, 118.776, 836.973, 0.47174, 0.226968, 0.121181, 0.0691863, 83, 0.533914, 0.304829, 0.570932])
out.append([8.94508e-12, 0.999943, 5.69827e-05])

test.append([332.566, -0.455782, 0.2891, 116.619, 385.761, 0.354017, 0.370331, 0.200373, 0.117434, 52, 0.541063, 0.317105, 0.586078])
out.append([1.31571e-07, 0.996561, 0.00343849])

test.append([320.904, -0.421108, -2.04037, 101.786, 364.289, 0.571057, 0.218681, 0.0807715, 0.0608129, 49, 0.369357, 0.27809, 0.752901])
out.append([5.32527e-07, 0.994435, 0.00556478])

test.append([336.765, -0.0299227, -1.05511, 96.4468, 350.449, 0.137179, 0.199871, 0.100784, 0.0841615, 51, 0.504243, 0.421079, 0.835072])
out.append([6.53911e-07, 0.978998, 0.0210015])

test.append([373.198, -1.16073, -0.161898, 155.802, 672.425, 0.820147, 0.46746, 0.172997, 0.13616, 62, 0.370078, 0.291276, 0.787067])
out.append([3.72715e-08, 0.996135, 0.00386539])

test.append([330.286, 0.905652, 2.14171, 64.7321, 479.641, 0.272769, 0.14422, 0.111482, 0.0952928, 75, 0.772997, 0.660745, 0.854784])
out.append([1.6459e-11, 0.999954, 4.65035e-05])

test.append([382.51, -0.874635, -1.52453, 43.4994, 540.14, 0.46891, 0.038883, 0.02711, 0.0212291, 45, 0.69722, 0.545972, 0.78307])
out.append([3.76993e-06, 0.98562, 0.0143766])

test.append([314.764, -0.959247, 1.65701, 112.518, 484.283, 0.583614, 0.396895, 0.147521, 0.0859604, 95, 0.371688, 0.216582, 0.582698])
out.append([7.84214e-12, 0.957958, 0.042042])

test.append([327.193, -0.825145, -2.67121, 88.9911, 453.855, 0.184723, 0.246193, 0.115651, 0.0638349, 72, 0.469758, 0.259288, 0.551961])
out.append([1.79404e-10, 0.999708, 0.000291798])

test.append([496.622, 0.314913, 0.43837, 144.759, 541.172, 0.246346, 0.274963, 0.180745, 0.0887648, 67, 0.657344, 0.322824, 0.491104])
out.append([2.41397e-10, 0.99991, 8.99885e-05])

test.append([372.487, 0.875108, -1.66709, 93.0346, 532.64, 0.405905, 0.222862, 0.0962802, 0.0803607, 96, 0.432017, 0.360585, 0.834655])
out.append([1.3054e-11, 0.996101, 0.00389921])

test.append([326.474, -1.56467, -2.87136, 70.2333, 817.611, 0.525389, 0.2364, 0.0723046, 0.0479088, 48, 0.305857, 0.20266, 0.662597])
out.append([1.01931e-06, 0.994913, 0.00508595])

test = np.array(test)
out = np.array(out)
test *= stds
test += means
model.predict(test)/out


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
# NN.Shape()
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
