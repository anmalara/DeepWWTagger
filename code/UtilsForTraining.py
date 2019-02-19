import numpy as np
from math import *
import sys
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau

from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

nbins = 100


colorsample = {"Other Bkg" : "k", "Random" : "r",
               "Higgs": "tab:red", "QCD": "b", "Top": "tab:green", "DY": "tab:olive", "WJets": "m", "WZ": "tab:purple", "ZZ": "tab:cyan",
               "MC_HZ": "tab:red", "MC_QCD": "b", "MC_TTbar": "tab:green", "MC_DYJets": "tab:olive", "MC_WZ": "m", "MC_ZZ": "tab:cyan"}

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts))
        else:
            print "%r  %2.2f s" % \
                  (method.__name__, (te - ts))
        return result
    return timed


@timeit
def NNResponce(labels, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name = "NNResponce"):
    for i_signal, signal in enumerate(sample_names):
        for i_bkg, bkg in enumerate(sample_names):
            if i_signal == i_bkg:
                continue
            mask = np.any(labels[:,tuple(sorted((i_signal, i_bkg)))], axis=1)
            pred = predictions[mask][:,i_signal]
            lab = labels[mask][:,i_signal]
            otherbkg = predictions[~mask][:,i_signal]
            fpr, tpr, thr = roc_curve(lab, pred)
            plt.cla()
            plt.xticks( np.arange(0.1,1.1,0.1) )
            plt.grid(True, which="both")
            plt.xlabel("NN responce")
            plt.ylabel("A.U.")
            plt.hist(pred[lab==1], alpha = 0.5, bins=nbins, log=True, label=signal, color = colorsample[signal])
            plt.hist(pred[lab!=1], alpha = 0.5, bins=nbins, log=True, label=bkg, color = colorsample[bkg])
            plt.hist(otherbkg, alpha = 0.5, bins=nbins, log=True, label="Other Bkg", color = colorsample["Other Bkg"])
            with open(name+"WP.txt","w") as f:
                f.write("fpr \t tpr \t thr\n")
                for wp in [10,1,0.1]:
                    index = (np.abs(fpr - wp/100.)).argmin()
                    plt.axvline(x=thr[index], color = "r", label = "@"+str(wp)+"%"+"fpr: sig_tpr="+str(round(tpr[index],2))+" allbkgs_fpr="+str(round(len(otherbkg[otherbkg>thr[index]])*1./len(otherbkg),3)) )
                    f.write(str(fpr[index])+"\t"+str(tpr[index])+"\t"+str(thr[index])+"\n")
            plt.legend(loc="lower center", shadow=True, title=signal+" vs "+bkg+"   auc = "+str(round(auc(fpr,tpr),3)))
            if save_figure:
                if isinstance(save_figure, bool):
                    plt.savefig(name+signal+"vs"+bkg+".png")
                    plt.savefig(name+signal+"vs"+bkg+".pdf")
            if show_figure:
                plt.show()






@timeit
def plot_outputs_1d(NN, isLogy=True, show_figure=True, save_figure=False, name = "Outputs"):
    classes = to_categorical(np.arange(len(NN.sample_names)))
    for i_cl, cl in enumerate(NN.sample_names):
        plt.cla()
        plt.xticks( np.arange(0.1,1.1,0.1) )
        plt.grid(True, which="both")
        plt.xlabel("NN responce")
        plt.ylabel("A.U.")
        for i_sample, sample in enumerate(NN.sample_names):
            mask_train = np.all(NN.labels_train[:]==classes[i_sample], axis=1)
            mask_val = np.all(NN.labels_val[:]==classes[i_sample], axis=1)
            train = NN.predictions_train[mask_train][:,i_cl]
            val = NN.predictions_val[mask_val][:,i_cl]
            y_val, _ = np.histogram(val, bins=nbins)
            y_test, _ = np.histogram(val, bins=nbins)
            y_train, bins_train, _ = plt.hist(train, alpha=0.5, bins=nbins, density=False, histtype="step", log=True, label="Training sample,"+sample, color = colorsample[sample])
            plt.errorbar(0.5*(bins_train[1:] + bins_train[:-1]), y_val*len(train)/len(val), yerr=y_val**0.5, fmt=".", label="Validation sample,"+sample, color=colorsample[sample])
        plt.legend(loc="lower center", shadow=True)
        if save_figure:
            if isinstance(save_figure, bool):
                plt.savefig(name+cl+".png")
                plt.savefig(name+cl+".pdf")
        if show_figure:
            plt.show()


@timeit
def MaximiseSensitivity(labels, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name = "Sensistivity"):
    with open(name+".txt","w") as f:
        f.write("signal \t bkg \t Sensistivity.max() \t tpr \t fpr \t thr\n")
        for i_signal, signal in enumerate(sample_names):
            for i_bkg, bkg in enumerate(sample_names):
                if i_signal == i_bkg:
                    continue
                mask = np.any(labels[:,tuple(sorted((i_signal, i_bkg)))], axis=1)
                pred = predictions[mask][:,i_signal]
                lab = labels[mask][:,i_signal]
                fpr, tpr, thr = roc_curve(lab, pred)
                mask = fpr>0.
                fpr = fpr[mask]
                tpr = tpr[mask]
                thr = thr[mask]
                Sensistivity = tpr/np.sqrt(tpr+100000*fpr)
                f.write(signal+" \t "+bkg+" \t "+str(Sensistivity.max())+" \t "+str(tpr[Sensistivity.argmax()])+" \t "+str(fpr[Sensistivity.argmax()])+" \t "+str(thr[Sensistivity.argmax()])+"\n")

@timeit
def IndexingMatrix(matrix, matrix_check, check):
    return matrix[np.asarray([np.array_equal(el,check) for el in matrix_check])]


@timeit
def plot_ROC_Curves(labels, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name = "ROC"):
    classes = to_categorical(np.arange(len(sample_names)))
    plt.cla()
    plt.xticks( np.arange(0.1,1.1,0.1) )
    plt.grid(True, which="both")
    for i, sample in enumerate(sample_names):
        fpr, tpr, thr = roc_curve(labels[:,i], predictions[:,i])
        label = sample+": auc = "+str(round(auc(fpr,tpr),3))
        # matrix_ = IndexingMatrix(predictions, labels, classes[i])
        # mean = matrix_.mean(axis=0)
        # std = matrix_.std(axis=0)
        # label = sample+": auc = "+str(round(auc(fpr,tpr),3))+", mean = ["
        # for j in range(len(sample_names)):
        #     label += str(round(mean[j],3))+"+-"+str(round(std[j],3))+","
        # label += "]"
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
    plt.legend(loc="best", shadow=True)
    if save_figure:
        if isinstance(save_figure, bool):
            plt.savefig(name+".png")
            plt.savefig(name+".pdf")
    if show_figure:
        plt.show()



@timeit
def plot_ROC_Curves1vs1(labels, predictions, sample_names, isLogy=True, show_figure=True, save_figure=False, name = "ROC"):
    for i_signal, signal in enumerate(sample_names):
        plt.cla()
        plt.xticks( np.arange(0.1,1.1,0.1) )
        plt.grid(True, which="both")
        mask = labels[:,sample_names.index(signal)] == 1
        for i_bkg, bkg in enumerate(sample_names):
            if bkg == signal:
                continue
            mask = np.any(labels[:,tuple(sorted((i_signal, i_bkg)))], axis=1)
            fpr, tpr, thr = roc_curve(labels[mask][:,i_signal], predictions[mask][:,i_signal])
            label = bkg+": auc = "+str(round(auc(fpr,tpr),3))
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
        plt.legend(loc="best", shadow=True)
        if save_figure:
            if isinstance(save_figure, bool):
                plt.savefig(name+"_"+signal+".png")
                plt.savefig(name+"_"+signal+".pdf")
        if show_figure:
            plt.show()



@timeit
def plot_losses(hist, show_figure=True, save_figure=False, losses="loss", min_epoch=0, name = "history" ):
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
            plt.savefig(name+".png")
            plt.savefig(name+".pdf")
    if show_figure:
        plt.show()

@timeit
def PlotInfos(NN, show_figure = False, save_figure = True):
    plot_ROC_Curves(NN.labels_val, NN.predictions_val, NN.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"Roc")
    plot_ROC_Curves1vs1(NN.labels_val, NN.predictions_val, NN.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"Roc1vs1")
    plot_outputs_1d(NN, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = NN.modelpath+"Outputs")
    NNResponce(NN.labels_val, NN.predictions_val, NN.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = NN.modelpath+"NNResponce")
    MaximiseSensitivity(NN.labels_val, NN.predictions_val, NN.sample_names, isLogy=True, show_figure=show_figure, save_figure=save_figure, name = NN.modelpath+"Sensistivity")
    plot_losses(NN.callbacks[0], min_epoch=0,  losses="loss", show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"loss")
    plot_losses(NN.callbacks[0], min_epoch=0,  losses="acc",  show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"acc")
    plot_losses(NN.callbacks[0], min_epoch=10, losses="loss", show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"loss_10")
    plot_losses(NN.callbacks[0], min_epoch=10, losses="acc",  show_figure=show_figure, save_figure=save_figure, name=NN.modelpath+"acc_10")



@timeit
def DefineCallbacks(modelpath):
    callbacks = []
    history = History()
    callbacks.append(history)
    modelCheckpoint_loss        = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor="val_loss", save_best_only=False)
    callbacks.append(modelCheckpoint_loss)
    modelCheckpoint_acc         = ModelCheckpoint(modelpath+"model_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor="val_acc", save_best_only=False)
    callbacks.append(modelCheckpoint_acc)
    modelCheckpoint_loss_best   = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_loss{val_loss:.2f}.h5", monitor="val_loss", save_best_only=True)
    callbacks.append(modelCheckpoint_loss_best)
    modelCheckpoint_acc_best    = ModelCheckpoint(modelpath+"bestmodel_epoch{epoch:03d}_acc{val_acc:.2f}.h5", monitor="val_acc", save_best_only=True)
    callbacks.append(modelCheckpoint_acc_best)
    reduceLROnPlateau           = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=0.001, cooldown=10)
    callbacks.append(reduceLROnPlateau)
    return callbacks
