from root_numpy import *
import numpy as np
import math
from math import *
import os.path
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.cla()
plt.figure()

fpr = np.load("Sequential/fpr_epoch_20_pt_300_500.npy")
tpr = np.load("Sequential/tpr_epoch_20_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 20, 300 < pt < 500')

fpr = np.load("Sequential/fpr_epoch_200_pt_300_500.npy")
tpr = np.load("Sequential/tpr_epoch_200_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 200, 300 < pt < 500')

fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_seq_300_500.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("Sequential/fpr_epoch_20_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_20_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 20, pt > 500')

fpr = np.load("Sequential/fpr_epoch_200_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_200_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 200, pt > 500')

fpr = np.load("Sequential/fpr_epoch_500_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_500_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 500, pt > 500')

fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')


plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_seq_500.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')

fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_seq.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("JetImage/fpr_epoch_20_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_20_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 20, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_50_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_50_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 50, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_100_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_100_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 100, 300 < pt < 500')

# fpr = np.load("JetImage/fpr_epoch_200_pt_300_500.npy")
# tpr = np.load("JetImage/tpr_epoch_200_pt_300_500.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 200, 300 < pt < 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_JetImage_300_500.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("JetImage/fpr_epoch_20_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_20_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 20, pt > 500')

fpr = np.load("JetImage/fpr_epoch_50_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_50_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 50, pt > 500')

fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')

fpr = np.load("JetImage/fpr_epoch_100_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_100_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 100, pt > 500')

# fpr = np.load("JetImage/fpr_epoch_200_pt_500_10000.npy")
# tpr = np.load("JetImage/tpr_epoch_200_pt_500_10000.npy")
# plt.semilogy(tpr, fpr, label='JetImage epochs = 200, pt > 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_JetImage_500.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_100_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_100_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 100, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')

fpr = np.load("JetImage/fpr_epoch_100_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_100_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 100, pt > 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN_JetImage.png")

################################################################################
################################################################################

plt.cla()
plt.figure()

fpr = np.load("JetImage/fpr_epoch_70_pt_300_500.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, 300 < pt < 500')

fpr = np.load("Sequential/fpr_epoch_2000_pt_300_500.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_300_500.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, 300 < pt < 500')

fpr = np.load("JetImage/fpr_epoch_70_pt_500_10000.npy")
tpr = np.load("JetImage/tpr_epoch_70_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='JetImage epochs = 70, pt > 500')

fpr = np.load("Sequential/fpr_epoch_2000_pt_500_10000.npy")
tpr = np.load("Sequential/tpr_epoch_2000_pt_500_10000.npy")
plt.semilogy(tpr, fpr, label='Sequential epochs = 2000, pt > 500')

plt.semilogy([0, 1], [0, 1], 'k--')
plt.xticks( [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] )
plt.grid(True, which='both')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.001, 1.05])
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='upper left', shadow=True)
# plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.savefig("transfer/NN.png")
