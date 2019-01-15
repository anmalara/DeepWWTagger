import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from UtilsForTraining import *
from variables import *

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
        else: return "MaxAbsScaler"
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

def SequentialModel(optimizer="adam", kernel_initializer="glorot_normal", bias_initializer="glorot_normal", activation="relu",dropoutRate=0.1):
    layers = {"input": [data_train.shape[1],50], "layers": [50,50,50,50,50,50,50,10], "output": [labels_train.shape[1]]}
    params = {"activation_last": "softmax", "metrics":["accuracy"]}
    model = Sequential()
    # Define layers
    model.add(Dense(layers["input"][1], input_dim=layers["input"][0], activation=activation,kernel_initializer=kernel_initializer))
    for i in range(0,len(layers["layers"])):
        model.add(Dense(layers["layers"][i], activation=activation,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
        model.add(BatchNormalization())
        model.add(Dropout(dropoutRate))
    model.add(Dense(layers["output"][0], activation=params["activation_last"]))
    # Compile
    if layers["output"][0]==1:
        myloss = "binary_crossentropy"
    else:
        myloss = "categorical_crossentropy"
    model.compile(loss=myloss, optimizer=optimizer, metrics=params["metrics"])
    model.summary()
    # TODO
    # save model params
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
    return model

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

try:
    info = sys.argv[1]
    radius = sys.argv[2]
    pt_min = int(sys.argv[3])
    pt_max = int(sys.argv[4])
except:
    info = "JetInfo"
    radius = "AK8"
    pt_min = 300
    pt_max = 500

print info, radius, pt_min, pt_max
filePath = "/beegfs/desy/user/amalara/"
# filePath = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"
# filePath = "/Users/andrea/Desktop/Analysis/"

inputFolder = filePath+"input_varariables/NTuples_Tagger/Sequential/"
modelName = "model_"+info+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+"/"
print modelName
outputFolder = filePath+"output_varariables/Sequential/"+modelName
modelpath = outputFolder+"models_Grid/"

if not os.path.exists(modelpath):
    os.makedirs(modelpath)

isSubset = True
sample_names = sorted(["Higgs", "QCD", "Top","DY"])
isGen = 0
if "Gen" in info:
    isGen = 1

data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names,max_size=500000)

Normalization(data_train,[data_train,data_val,data_test])


if isSubset:
    if isGen:
        branch = branch_names_dict["GenJetInfo"]
        subset = (branch.index("GenSoftDropMass"), branch.index("GenJetTau1"), branch.index("GenJetTau2"), branch.index("GenJetTau3"), branch.index("GenJetTau4"))
    else:
        branch = branch_names_dict["JetInfo"]
        subset = (branch.index("jetMassSoftDrop"), branch.index("jetTau1"), branch.index("jetTau2"), branch.index("jetTau3"), branch.index("jetTau4"), branch.index("jetBtag"))
    subset = sorted(set(subset))
    data_train = data_train[:,subset]
    data_val   = data_val[:,subset]
    data_test  = data_test[:,subset]



from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=SequentialModel, verbose=0)


# layers = {"input": [data_train.shape[1],50], "layers": [50,50,50,50,50,50,50,10], "output": [labels_train.shape[1]]}
# params = {"activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01}
# params["batch_size"] = 128
# params["epochs"] = 3


# define the grid search parameters
batch_size = [10, 128, 1024, 4096]
epochs = [3]
optimizer = ['SGD','Adam']
dropoutRate = [0.001, 0.01, 0.2]
kernel_initializer = ['lecun_uniform','zero', 'glorot_normal']
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']



batch_size = [10]
epochs = [3]
optimizer = ['Adam']
dropoutRate = [0.001]
kernel_initializer = ['glorot_normal']
activation = ['relu',]



param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, kernel_initializer=kernel_initializer, bias_initializer=kernel_initializer, activation=activation, dropoutRate=dropoutRate)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(data_train, labels_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


quit()



# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = KerasClassifier(build_fn=create_model, verbose=0)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
