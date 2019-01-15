import numpy as np
from math import sqrt, fabs
from variables import *

def CreateJetVariables(Vars, JetInfo="JetInfo", CandInfo="CandInfo"):
    jet_variables = []
    for event in range(Vars[CandInfo].shape[0]):
        px, py, pz, pt, E  = 0, 0, 0, 0, 0
        for cand in range(Vars[CandInfo].shape[2]):
            weight = Vars[CandInfo][event,CandPuppiWeight_index,cand]
            px += Vars[CandInfo][event,CandPx_index,cand]*weight
            py += Vars[CandInfo][event,CandPy_index,cand]*weight
            pz += Vars[CandInfo][event,CandPz_index,cand]*weight
            pt += Vars[CandInfo][event,CandPt_index,cand]*weight
            E  += Vars[CandInfo][event,CandEnergy_index,cand]*weight
        vars = np.array((px,py,pz,pt,sqrt(px*px+py*py),E,sqrt(fabs(E*E-px*px-py*py-pz*pz))))
        vars = np.expand_dims(vars, axis=0)
        jet_variables.append(vars)
    if len(jet_variables)>0:
        jet_variables = np.concatenate(jet_variables)
    else:
        jet_variables = np.empty((Vars[CandInfo].shape[0],6))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau2_index]/Vars[JetInfo][:,jetTau1_index], axis=1)))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau3_index]/Vars[JetInfo][:,jetTau1_index], axis=1)))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau4_index]/Vars[JetInfo][:,jetTau1_index], axis=1)))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau3_index]/Vars[JetInfo][:,jetTau2_index], axis=1)))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau4_index]/Vars[JetInfo][:,jetTau2_index], axis=1)))
    jet_variables = np.hstack((jet_variables, np.expand_dims(Vars[JetInfo][:,jetTau4_index]/Vars[JetInfo][:,jetTau3_index], axis=1)))
    return jet_variables
