import sys
from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D
from keras import regularizers
from keras.models import Model, load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils, generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Dropout
from keras.models import model_from_yaml
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


sys.path.append("/home/amalara/DeepWWTagger/code/../LorentzLayer")

from cola import CoLa
from lola import LoLa

seed = 7
np.random.seed(seed)

ncand = 40
selection_cuts = "((abs(jetEta)<=2.7)&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1))||(abs(jetEta)<2.4&&(jetNhf<0.9)&&(jetPhf<0.9)&&(jetMuf<0.80)&&(ncandidates>1)&&(jetChf>0)&&(jetChm>0)&&(jetElf<0.8))"
jet_branch_names = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3"]
pf_branch_names = ["candEnergy", "candPx", "candPy", "candPz", "candPt", "candEta", "candPhi", "candPdgId", "candMass", "candDXY", "candDZ", "candPuppiWeight"]




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
    # plt.title('ROC curve (area = %0.2f)' % roc_auc)
    # plt.savefig("Sequential/"+radius+"/Roc_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+".png")
    plt.savefig(name_fig)



def model_lola(params):
   model = Sequential()
   model.add(CoLa(input_shape = (params["n_features"], params["n_constit"]),
                  add_total   = False,
                  add_eye     = True,
                  debug =  False,
                  n_out_particles = params["n_out_particles"]))
   model.add(LoLa(
      input_shape = (params["n_features"], params["n_constit"]+ params["n_out_particles"]),
      debug = False,
      train_metric = False,
      es  = 1,
      xs  = 1,
      ys  = 1,
      zs  = 1,
      cs  = 0,
      vxs = 0,
      vys = 0,
      vzs = 0,
      ms  = 1,
      pts = 1,
      n_train_es  = 1,
      n_train_ms  = 1,
      n_train_pts = 1,
      n_train_sum_dijs   = 2,
      n_train_min_dijs   = 2))
   model.add(Flatten())
   # model.add(Flatten(input_shape = (params["n_features"], params["n_constit"])))
   # model.add(Dense(160, input_dim=160, activation='relu',kernel_initializer='lecun_uniform'))
   model.add(Dense(100))
   model.add(Activation('relu'))
   model.add(Dense(50))
   model.add(Activation('relu'))
   model.add(Dense(params["n_classes"], activation='softmax'))
   return model




tpr_array = []
fpr_array = []
name_array = []

losses = []
accuracy = []
val_losses = []
val_accuracy = []


file_path = "/beegfs/desy/user/amalara/Ntuples/test/"


import pandas
radius = "AK8"
pt_min = 550
pt_max = 650
ncand = 40
file_max = 1000000

input_filename = file_path+"train.h5"
store = pandas.HDFStore(input_filename)
temp = store.select("table",stop=file_max)
temp = temp.as_matrix()
data = temp[:,0:800]
info = temp[:,800:805]
label = temp[:,[805]]
label = np.concatenate((label, 1-label), axis=1)

file_max = 100000
input_filename = file_path+"test.h5"
store = pandas.HDFStore(input_filename)
temp = store.select("table",stop=file_max)
temp = temp.as_matrix()
data_test = temp[:,0:800]
info_test = temp[:,800:805]
label_test = temp[:,[805]]
label_test = np.concatenate((label_test, 1-label_test), axis=1)

input_filename = file_path+"val.h5"
store = pandas.HDFStore(input_filename)
temp = store.select("table",stop=file_max)
temp = temp.as_matrix()
data_val = temp[:,0:800]
info_val = temp[:,800:805]
label_val = temp[:,[805]]
label_val = np.concatenate((label_val, 1-label_val), axis=1)


data = data.reshape((data.shape[0],4, 200),order='F')
data_test = data_test.reshape((data_test.shape[0],4, 200),order='F')
data_val = data_val.reshape((data_val.shape[0],4, 200),order='F')

data = data[:,:,0:ncand]
data_test = data_test[:,:,0:ncand]
data_val = data_val[:,:,0:ncand]

print data.shape
print data_test.shape
print data_val.shape

print label.shape
print label_test.shape
print label_val.shape


params = {'n_features': 4, 'n_constit': ncand, 'n_out_particles': 15, 'n_classes': 2}
model = model_lola(params)
model.summary()

# model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0000000001), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

epoch = 10
print "trovato"
print data.shape
history = model.fit(data, label, batch_size=256, epochs=epoch, verbose=1, validation_data=(data_test,label_test))


out_file_path = "/beegfs/desy/user/amalara/output_varariables/test/"

losses.append(history.history['loss'][0])
accuracy.append(history.history['acc'][0])
val_losses.append(history.history['val_loss'][0])
val_accuracy.append(history.history['val_acc'][0])


predictions = model.predict(data_val)
print predictions.shape
fpr, tpr, thr = roc_curve(label_val[:,0], predictions[:,0])
roc_auc = auc(fpr, tpr)
print(roc_auc)

print fpr.shape, tpr.shape, thr.shape, predictions.shape
print fpr, tpr, thr

# print label
print predictions


model.save(out_file_path+"Lola/"+radius+"/model_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_event_"+str(file_max)+".h5")
np.save(out_file_path+"Lola/"+radius+"/fpr_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_event_"+str(file_max)+".npy", fpr)
np.save(out_file_path+"Lola/"+radius+"/tpr_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_event_"+str(file_max)+".npy", tpr)
np.save(out_file_path+"Lola/"+radius+"/thr_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_event_"+str(file_max)+".npy", thr)
np.save(out_file_path+"Lola/"+radius+"/pred_epoch_"+str(epoch)+"_pt_"+str(pt_min)+"_"+str(pt_max)+"_event_"+str(file_max)+".npy", predictions)


tpr_array.append(tpr)
fpr_array.append(fpr)
name_array.append(radius)

plot_ROC_Curve(tpr_array, fpr_array, name_array, out_file_path+"Lola/"+radius+"/ROC"+radius+"_event_"+str(file_max)+".png")


plt.cla()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(out_file_path+"Lola/"+radius+"/acc_"+radius+"_event_"+str(file_max)+".png")

# summarize history for loss
plt.cla()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(out_file_path+"Lola/"+radius+"/loss_"+radius+"_event_"+str(file_max)+".png")
