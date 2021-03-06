from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Conv2D, MaxPooling2D
from keras import regularizers
from keras.models import Model, load_model
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

seed = 7
np.random.seed(seed)
start_time=time.time()
temp_time = start_time

branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]
branch_names_gen_jet = ["GenJetPt", "GenJetEta", "GenJetPhi", "GenJetMass", "GenJetEnergy", "isBJetGen", "GenSoftDropMass", "GenJetTau1", "GenJetTau2", "GenJetTau3", "GenJetTau4"]


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


##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################


file_path = "/beegfs/desy/user/amalara/"
name_folder = "input_varariables/NTuples_Tagger/JetImage/"
name_folder_output = "output_varariables/JetImage/"
if not os.path.exists(file_path+name_folder_output):
    os.makedirs(file_path+name_folder_output)

file_min = int(sys.argv[1])
file_max = int(sys.argv[2])
pt_min = int(sys.argv[3])
pt_max = int(sys.argv[4])

name_variable = sys.argv[5]
radius =sys.argv[6]

if not os.path.exists(file_path+name_folder_output+radius):
    os.makedirs(file_path+name_folder_output+radius)

if name_variable=='norm':
    name_variable=""

if name_variable=='gen_':
    isGen = 1
else:
    isGen = 0

print file_min, file_max, pt_min, pt_max, name_variable, radius

bkg = "Higgs"
Higgs = np.load(file_path+name_folder+"JetImage_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy")
bkg = "QCD"
QCD = np.load(file_path+name_folder+"JetImage_"+name_variable+"input_variable_"+bkg+"_"+radius+"_file_"+str(file_min)+"_"+str(file_max)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy")

variable_used=name_variable

(Higgs, Higgs_test, Higgs_val) = np.array_split(Higgs, 3)
(QCD, QCD_test, QCD_val) = np.array_split(QCD, 3)
data = np.concatenate((Higgs, QCD))
data_test = np.concatenate((Higgs_test, QCD_test))
data_val = np.concatenate((Higgs_val, QCD_val))
label = np.concatenate((np.ones(Higgs.shape[0]),np.zeros(QCD.shape[0])))
label_test = np.concatenate((np.ones(Higgs_test.shape[0]),np.zeros(QCD_test.shape[0])))
label_val = np.concatenate((np.ones(Higgs_val.shape[0]),np.zeros(QCD_val.shape[0])))

tpr_array = []
fpr_array = []
name_array = []

losses = []
accuracy = []
val_losses = []
val_accuracy = []


input_shape = (Higgs.shape[1],Higgs.shape[2],Higgs.shape[3])
model = Sequential()
# model.add(Conv2D(192, kernel_size=(2,2), strides=(1, 1), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
# model.add(Dropout(0.1))
model.add(Conv2D(96, kernel_size=(2,2), strides=(1, 1), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
model.add(Dropout(0.1))
model.add(Conv2D(48, kernel_size=(2,2), strides=(1, 1), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
model.add(Dropout(0.1))
if "8" in radius:
    model.add(Conv2D(12, kernel_size=(2,2), strides=(2, 2), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
    model.add(Dropout(0.1))
    model.add(Conv2D(2, kernel_size=(2,2), strides=(1, 1), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
    model.add(Dropout(0.1))
if "15" in radius:
    model.add(Conv2D(12, kernel_size=(3,3), strides=(2, 2), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
    model.add(Dropout(0.1))
    model.add(Conv2D(2, kernel_size=(3,3), strides=(2, 2), padding="valid",input_shape=input_shape,activation="relu", data_format="channels_first"))
    model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(50, activation='relu', init='lecun_uniform'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu', init='lecun_uniform'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu', init='lecun_uniform'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid', init='lecun_uniform'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001),  metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])

epoch = 1
history = model.fit(data, label, batch_size=256, epochs=epoch, verbose=1, validation_data=(data_test,label_test))


losses.append(history.history['loss'][0])
accuracy.append(history.history['acc'][0])
val_losses.append(history.history['val_loss'][0])
val_accuracy.append(history.history['val_acc'][0])


predictions = model.predict(data_val)
fpr, tpr, thr = roc_curve(label_val, predictions)
roc_auc = auc(fpr, tpr)
print(roc_auc)

print predictions[:,0].shape
print fpr.shape, tpr.shape
print fpr

model.save(file_path+name_folder_output+radius+"/model_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".h5")
np.save(file_path+name_folder_output+radius+"/fpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", fpr)
np.save(file_path+name_folder_output+radius+"/tpr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", tpr)
np.save(file_path+name_folder_output+radius+"/thr_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", thr)
np.save(file_path+name_folder_output+radius+"/pred_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", predictions)
np.save(file_path+name_folder_output+radius+"/label_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", label_val)
np.save(file_path+name_folder_output+radius+"/losses_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", losses)
np.save(file_path+name_folder_output+radius+"/accuracy_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", accuracy)
np.save(file_path+name_folder_output+radius+"/val_losses_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", val_losses)
np.save(file_path+name_folder_output+radius+"/val_accuracy_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".npy", val_accuracy)


tpr_array.append(tpr)
fpr_array.append(fpr)
name_array.append(radius)

plot_ROC_Curve(tpr_array, fpr_array, name_array, file_path+name_folder_output+radius+"/ROC_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".png")


plt.cla()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(file_path+name_folder_output+radius+"/acc_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".png")

# summarize history for loss
plt.cla()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(file_path+name_folder_output+radius+"/loss_var_"+variable_used+"_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".png")
