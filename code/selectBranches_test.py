from root_numpy import root2array
from root_numpy import array2root
from root_numpy import rec2array

selection_cuts = "(abs(jetEta)<=2.7)"
branch_names_jet = ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetMassSoftDrop", "jetTau1", "jetTau2", "jetTau3", "jetTau4"]


file_path = "/beegfs/desy/user/amalara/"
name_folder = "Ntuples/MC-CandidatesP8-Higgs-GenInfos-v3/180410_123725/"
name_variable = "flatTreeFileHiggs-nano_"
file_name = file_path+name_folder+"0000/"+name_variable+str(1)+".root"
radius = "AK8"
tree_name = "boosted"+radius+"/events"
branch_names = branch_names_jet
final_file = 'test.root'

file = root2array(filenames=file_name, treename=tree_name, branches=branch_names, selection=selection_cuts)
file = rec2array(file)
array2root(file, final_file, mode='recreate')
