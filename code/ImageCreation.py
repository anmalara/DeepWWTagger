import numpy as np

from variables import *


def CreateMatrix(pf, eta_jet, phi_jet, Radius):
    n_eta = int((2*Radius)/step_eta)
    n_phi = int((2*Radius)/step_phi)
    #created the minimum edges on the 2 axes
    matrix = np.zeros((n_images,n_eta,n_phi))
    pt_pf = pf[CandPt_index,:]
    eta_pf = pf[CandEta_index,:]
    phi_pf = pf[CandPhi_index,:]
    pdgId_pf = np.absolute(pf[CandPdgId_index,:]).astype(np.int)
    mask = (np.absolute(eta_pf)<Radius)*(np.absolute(phi_pf)<Radius)
    pt_pf = pt_pf[mask]
    eta_pf = eta_pf[mask]
    phi_pf = phi_pf[mask]
    pdgId_pf = pdgId_pf[mask]
    for i in range(len(pt_pf)):
        x = int((eta_pf[i]+Radius)//step_eta)
        y = int((phi_pf[i]+Radius)//step_phi)
        if pdgId_pf[i] == 130: #neutral hadron
            matrix[0,x,y] += 1
        elif pdgId_pf[i] == 211: #charged hadron
            matrix[1,x,y] += 1
            matrix[2,x,y] += pt_pf[i]/np.cosh(eta_pf[i])
            # matrix[2,x,y] += pt_pf[i]
    return matrix

def CreateImage(Vars, Radius, CandInfo="CandInfo", JetInfo="JetInfo"):
    jet_images = []
    for i in range(Vars[CandInfo].shape[0]):
        eta_jet = Vars[JetInfo][i,jetEta_index]
        phi_jet = Vars[JetInfo][i,jetPhi_index]
        jet_image = CreateMatrix(Vars[CandInfo][i,:,:], eta_jet, phi_jet, Radius)
        jet_image = np.expand_dims(jet_image, axis=0)
        jet_images.append(jet_image)
    if len(jet_images)>0:
        jet_images = np.concatenate(jet_images)
    else:
        jet_images = np.empty((n_images,int((2*Radius)/step_eta),int((2*Radius)/step_phi)))
    # numpy.ndarray (n_events, n_images, 2*Radius, 2*Radius )
    return jet_images.astype(variable_type)
