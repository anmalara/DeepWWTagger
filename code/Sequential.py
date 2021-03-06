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
            "modelpath" : "model_newarch",
            "varnames" : ["JetInfo", "JetVariables"],
            # "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "ncandidates", "jetBtag", "jetTau1", "jetTau2", "jetTau3", "jetTau21", "jetTau31", "jetTau32"],
            "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetBtag", "jetTau1", "jetTau2"],
            "sample_names": sorted(["Higgs", "QCD", "Top" ]),
            "radius" : "AK8",
            "pt_min" : 300,
            "pt_max" : 500,
            "max_size" : -1,
            "layers" : [50,200,200,50,10],
            "params" : {"epochs" : 10, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
            "seed" : 4444
            }

np.random.seed(dict_var["seed"])

# NN = SequentialNN(dict_var)
# NN.InputShape()
# NN.CreateSubSet()
# NN.Normalization()




NN.FitModel()
NN.Predict()
NN.Plots(show_figure = False, save_figure = True)
NN.SaveModel()

quit()

NN.FitModel()
for i in range(10):
    add_layer(NN,nodes=[200,50,10],NodeToRemove=7)
    NN.FitModel()


def add_layer(NN, nodes, NodeToRemove=7):
    for i in range(NodeToRemove):
        NN.model.pop()
    for layer in NN.model.layers:
        layer.trainable = False
    for node in nodes:
        NN.model.add(Dense(node, activation=NN.params["activation"],kernel_initializer=NN.params["kernel_initializer"],bias_initializer=NN.params["bias_initializer"]))
        NN.model.add(BatchNormalization())
        NN.model.add(Dropout(NN.params["dropoutRate"]))
    NN.model.add(Dense(NN.labels_train.shape[1], activation=NN.params["activation_last"]))
    NN.model.compile(loss=NN.myloss, optimizer=NN.params["optimizer"], metrics=NN.params["metrics"])
    NN.model.summary()
