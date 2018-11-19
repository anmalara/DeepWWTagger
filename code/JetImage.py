import numpy as np
from math import *
import os.path
import os
import sys
import time
import copy

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from variables import *

seed = 0
np.random.seed(seed)

from root_numpy import *

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print '%r  %2.2f s' % \
                  (method.__name__, (te - ts))
        return result
    return timed

def JetImageModel(layers = {"input": [5,20], "layers": [10,10,10], "output": [1]}, params = {"activation" : "relu"}):
    model = Sequential()
    # Define layers
    # model.add(Conv2D(layers["layers"][0], kernel_size=(layers["kernel_size"][0],layers["kernel_size"][0]), strides=(layers["strides"][0],layers["strides"][0]), padding=params["padding"],input_shape=params["input_shape"], activation=params["activation"], data_format=params["data_format"]))
    model.add(Conv2D(layers["layers"][0], kernel_size=layers["kernel_size"][0], strides=layers["strides"][0], padding=params["padding"],input_shape=params["input_shape"], activation=params["activation"], data_format=params["data_format"]))
    for i in range(1,len(layers["layers"])):
        model.add(Conv2D(layers["layers"][i], kernel_size=layers["kernel_size"][i], strides=layers["strides"][i], padding=params["padding"], activation=params["activation"], data_format=params["data_format"]))
        # model.add(Conv2D(layers["layers"][i], kernel_size=(layers["kernel_size"][i],layers["kernel_size"][i]), strides=(layers["strides"][i],layers["strides"][i]), padding=params["padding"],input_shape=params["input_shape"], activation=params["activation"], data_format=params["data_format"]))
        # model.add(BatchNormalization())
        model.add(Dropout(params["dropoutRate"]))
    model.add(Flatten())
    for i in range(0,len(layers["Dense"])):
        model.add(Dense(layers["Dense"][i], activation=params["activation"],kernel_initializer=params["kernel_initializer"],bias_initializer=params["bias_initializer"]))
        # model.add(BatchNormalization())
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


@timeit
def InputShape_old(sample_names=["Higgs","QCD", "Top"],max_size=500000):
    first_sample = True
    for i, sample_name in enumerate(sample_names):
        first_file = True
        first_conc = True
        suncounter = 0
        for index in range(1,files_dictionary[sample_name][0]):
            try:
                temp = np.load(inputFolder+sample_name+"_"+radius+"/JetImage_"+info+"_"+sample_name+"_"+radius+"_file_"+str(index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", encoding = "bytes")
            except:
                continue
            if first_file:
                sample_temp = temp
                first_file = False
            else:
                sample_temp = np.concatenate((sample_temp, temp), axis=0)
                suncounter += temp.shape[0]
                if suncounter > 100000:
                    suncounter = 0
                    first_file = True
                    if first_conc:
                        sample = sample_temp
                        first_conc = False
                    else:
                        sample = np.concatenate((sample,sample_temp), axis=0)
                        print sample.shape
                        if sample.shape[0]>max_size:
                            break
        if suncounter !=0:
            sample = np.concatenate((sample,sample_temp))
        print(sample_name, sample.shape, index)
        label = np.ones(sample.shape[0],dtype=int)*i
        if first_sample:
            data = sample
            labels = label
            first_sample = False
        else:
            data = np.concatenate((data, sample))
            labels = np.concatenate((labels, label))
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, random_state=42, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test




@timeit
def InputShape(sample_names=["Higgs","QCD", "Top"],max_size=500000):
    data = []
    labels = []
    for i, sample_name in enumerate(sample_names):
        subcounter = 0
        sample = []
        for index in range(1,files_dictionary[sample_name][0]):
            try:
                temp = np.load(inputFolder+sample_name+"_"+radius+"/JetImage_"+info+"_"+sample_name+"_"+radius+"_file_"+str(index)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", encoding = "bytes")
            except:
                continue
            sample.append(temp)
            subcounter += temp.shape[0]
            if subcounter > max_size:
                break
        sample = np.concatenate(sample)
        print(sample_name, sample.shape, index)
        label = np.ones(sample.shape[0],dtype=int)*i
        data.append(sample)
        labels.append(label)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    labels = to_categorical(labels,num_classes=len(sample_names))
    data_train, data_val, labels_train, labels_val = train_test_split(data, labels, random_state=42, train_size=0.67)
    data_val, data_test, labels_val, labels_test = train_test_split(data_val, labels_val, random_state=42, train_size=0.5)
    return data_train, data_val, data_test, labels_train, labels_val, labels_test



def plot_ROC_Curve(tpr, fpr, name_plot, name_fig):
    plt.cla()
    plt.figure()
    plt.semilogy([0, 1], [0, 1], 'k--')
    plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
    plt.grid(True, which='both')
    # plt.semilogy(tpr, fpr, label='ROC curve')
    for i in range(0,len(tpr)):
        plt.semilogy(tpr[i], fpr[i], label=name_plot[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.001, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc='best', shadow=True)
    plt.title('ROC curve (area = %0.2f)' % roc_auc)
    # plt.savefig("Sequential/"+radius+"/Roc_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".png")
    plt.savefig(name_fig)



def IndexingMatrix(matrix, matrix_check, check):
    return matrix[np.asarray([np.array_equal(el,check) for el in matrix_check])]


def plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name = "history.png"):
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
    if save_figure:
        if isinstance(save_figure, bool):
            plt.savefig(name)
    if show_figure:
        plt.show()


def plot_losses(hist, show_figure=True, save_figure=False, losses="loss", min_epoch=0, name = "history.png" ):
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
            plt.savefig(name)
    if show_figure:
        plt.show()



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

inputFolder = filePath+"input_varariables/NTuples_Tagger/JetImage/"
outputFolder = filePath+"output_varariables/JetImage/"+modelName
modelpath = outputFolder+"models/"


data_train, data_val, data_test, labels_train, labels_val, labels_test = InputShape(sample_names,max_size=10000000)


input_shape = (data_train.shape[1],data_train.shape[2],data_train.shape[3])


layers = {"Dense": [40,30,10], "layers": [48,12,2], "kernel_size": [(2,2),(3,3),(3,3)], "strides": [(1,1),(2,2),(2,2)], "output": [labels_train.shape[1]]}
params = {"activation" : "relu", "input_shape" : input_shape, "padding" : "valid", "data_format" : "channels_first", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01}
params["batch_size"] = 100
params["epochs"] = 10

seed = 4444
np.random.seed(seed)
model = JetImageModel(layers, params)

callbacks = []
history = History()
callbacks.append(history)
# modelCheckpoint_loss      = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor='val_loss', save_best_only=False)
# callbacks.append(modelCheckpoint_loss)
# modelCheckpoint_acc       = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor='val_acc', save_best_only=False)
# callbacks.append(modelCheckpoint_acc)
# modelCheckpoint_loss_best = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor='val_loss', save_best_only=True)
# callbacks.append(modelCheckpoint_loss_best)
# modelCheckpoint_acc_best  = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor='val_acc', save_best_only=True)
# callbacks.append(modelCheckpoint_acc_best)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_delta=0.01, min_lr=0.001, cooldown=10)
# reduceLROnPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.001, cooldown=10)
# callbacks.append(reduceLROnPlateau)

model.fit(data_train, labels_train, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1, validation_data=(data_val,labels_val), callbacks=callbacks)

predictions = model.predict(data_test)


# plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True)

plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True, show_figure=False, save_figure=True, name="Roc.png")
plot_ROC_Curves(labels_test, predictions, sample_names, isLogy=True, show_figure=False, save_figure=True, name="Roc.pdf")


plot_losses(history, min_epoch=0, losses="loss", show_figure=False, save_figure=True, name="loss.png")
plot_losses(history, min_epoch=0, losses="acc", show_figure=False, save_figure=True, name="acc.png")
plot_losses(history, min_epoch=0, losses="loss", show_figure=False, save_figure=True, name="loss.pdf")
plot_losses(history, min_epoch=0, losses="acc", show_figure=False, save_figure=True, name="acc.pdf")
# plot_losses(history, min_epoch=10, losses="loss")
# plot_losses(history, min_epoch=10, losses="acc")


quit()

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])
