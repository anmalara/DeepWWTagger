import numpy as np
from math import *
import os.path
import os
import sys

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from variables import *


def loadVariables(bkg):
    temp = inputFolder+"Sequential_"+info+"_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy"
    return np.load(temp, encoding = "bytes")

def InputShape(sample_names=["Higgs","QCD", "Top"], subset = None):
    First = True
    for i, sample_name in enumerate(sample_names):
        sample = loadVariables(sample_name)
        if not isinstance(subset,tuple):
            subset = tuple(np.arange(0,sample.shape[1]))
        else:
            subset = sorted(set(subset))
            if subset[-1]>sample.shape[1]:
                subset = tuple(np.arange(0,sample.shape[1]))
        sample = sample[:,subset]
        label = np.ones(sample.shape[0],dtype=int)*i
        if First:
            data = sample
            labels = label
            First = False
        else:
            data = np.concatenate((data, sample))
            labels = np.concatenate((labels, label))
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test


seed = 0
np.random.seed(seed)

file_min = 0
file_max = 1000
pt_min = 500
pt_max = 10000
info = "JetInfo"
radius = "AK8"
isSubset = False
sample_names = ["Higgs", "QCD", "Top"]

modelName = "model_"+info+"_"+radius+"_pt_"+str(pt_min)+"_"+str(pt_max)+"/"
filePath = "/beegfs/desy/user/amalara/"
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


data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names, subset)


def SequentialModel(activation="relu", kernel_initializer="lecun_uniform", bias_initializer="zeros", activation_last="sigmoid", metrics=["accuracy"], optimizer="adam"):
    layers = {"input": [data_train.shape[1],20], "layers": [100,100,50,10], "output": [labels_train.shape[1]]}
    model = Sequential()
    # Define layers
    model.add(Dense(layers["input"][1], input_dim=layers["input"][0], activation=activation,kernel_initializer=kernel_initializer))
    for i in range(0,len(layers["layers"])):
        model.add(Dense(layers["layers"][i], activation=activation,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
    model.add(Dense(layers["output"][0], activation=activation_last))
    # Compile
    if layers["output"][0]==1:
        myloss = "binary_crossentropy"
    else:
        myloss = "categorical_crossentropy"
    model.compile(loss=myloss, optimizer=optimizer, metrics=metrics)
    # model.summary()
    # TODO
    # save model params
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, rankdir='LR')
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=SequentialModel, verbose=0)


# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [10, 50, 100]
optimizer = ['SGD','Adam']
learn_rate = [0.001, 0.01, 0.2]
kernel_initializer = ['lecun_uniform','zero', 'glorot_normal']
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']


batch_size = [10, 20, 40]
epochs = [10, 50, 100]
optimizer = ['SGD','Adam']
learn_rate = [0.001, 0.000001, 0.1]
kernel_initializer = ['lecun_uniform','ones', 'glorot_normal']
activation = ['softmax', 'relu']

data_train_ = data_train[:100000,:]
labels_train_ = labels_train[:100000,:]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, kernel_initializer=kernel_initializer, bias_initializer=kernel_initializer, activation=activation, activation_last=activation)
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(data_train_, labels_train_)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

data_train_ = data_train[:100000,:]
labels_train_ = labels_train[:100000,:]

data_train_ = data_train[:1000,:]
labels_train_ = labels_train[:1000,:]

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
