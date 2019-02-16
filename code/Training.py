from Models import *

##################################################
#                                                #
#                   MAIN Program                 #
#                                                #
##################################################

filePath = "/beegfs/desy/user/amalara/"
# filePath = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/"
# filePath = "/Users/andrea/Desktop/Analysis/"


dict_var = {"Info_dict": Info_dict,
            "isSubset" : True,
            "isGen" : False,
            "filePath" : "/beegfs/desy/user/amalara/",
            "modelpath" : "model_weights",
            "varnames" : ["JetInfo", "JetVariables"],
            # "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "ncandidates", "jetBtag", "jetTau1", "jetTau2", "jetTau3", "jetTau21", "jetTau31", "jetTau32"],
            "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetTau1", "jetTau2"],
            "sample_names": sorted(["Higgs", "QCD", "Top" ]),
            # "weights" : {"Higgs": 0.0003, "QCD": 2022100000, "Top": 313.9 },
            "weights" : {"Higgs": 1, "QCD": 1, "Top": 1 },
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : -1,
            "layers" : [50,200,200,50,10],
            "params" : {"epochs" : 10, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

NN = SequentialNN(dict_var)
NN.InputShape()
NN.CreateSubSet()
NN.Normalization()
NN.SequentialModel()
NN.FitModel()
NN.Predict()
NN.Plots(show_figure = False, save_figure = True)
NN.SaveModel()

quit()
