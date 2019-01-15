from variables import *
from math import pi as PI
import numpy as np
from numpy import linalg as LA

def preprocessing_ptSelection(Vars, pt_min, pt_max, JetInfo="JetInfo", Pt_index=jetPt_index ):
    mask = (Vars[JetInfo][:,Pt_index]>pt_min)*(Vars[JetInfo][:,Pt_index]<pt_max)
    for var in Vars:
        Vars[var] = Vars[var][mask]

def preprocessing_WMatchingSelection(Vars, JetInfo="JetExtraInfo", vars=[WMinusLep_index,WPlusLep_index]):
    mask = np.prod(Vars[JetInfo][:,vars]==[0,0], axis=1).astype(np.bool)
    for var in Vars:
        Vars[var] = Vars[var][mask]



def preprocessing_JetRotation(Vars, CandInfo="CandInfo"):
    eta = Vars[CandInfo][:, CandEta_index, :]
    phi = Vars[CandInfo][:, CandPhi_index, :]
    pt  = Vars[CandInfo][:, CandEnergy_index, :]
    eta -= np.expand_dims(np.sum(eta*pt,axis=1)/np.sum(pt,axis=1),axis=1)
    phi -= np.expand_dims(np.sum(phi*pt,axis=1)/np.sum(pt,axis=1),axis=1)
    I = np.array([[np.sum(phi*phi*pt,axis=1), np.sum(-phi*eta*pt,axis=1)], [np.sum(-phi*eta*pt,axis=1), np.sum(eta*eta*pt,axis=1)]]).swapaxes(0,1).swapaxes(0,2).astype(np.float32)
    R = LA.eigh(I)[1]
    angles = np.array((eta,phi)).swapaxes(0,1).swapaxes(1,2)
    angles = np.matmul(angles, R)
    Vars[CandInfo][:, CandEta_index, :] = angles[:,:,0]
    Vars[CandInfo][:, CandPhi_index, :] = angles[:,:,1]


def Preprocessing(Vars,pt_min, pt_max):
    preprocessing_ptSelection(Vars,pt_min, pt_max)
    preprocessing_WMatchingSelection(Vars)
    preprocessing_JetRotation(Vars)
