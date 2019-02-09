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
            "layers" : [50,200,200,200,200,200,200,50,10],
            "params" : {"epochs" : 5, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
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

#
#
#
#layers = {"input": [data_train.shape[1],50], "layers": [50,50,50,50,50,50,50,10], "output": [labels_train.shape[1]]}
#
#i=0
#
#par_list = []
#auc_list = []
#
#for batch_size in [10, 128, 1024, 4096]:
#    for epochs in [2]:
#        for optimizer in ['SGD','Adam',"adam"]:
#            for dropoutRate in [0.001, 0.01, 0.2]:
#                for kernel_initializer in ['lecun_uniform','zero', 'glorot_normal']:
#                    for bias_initializer in ['lecun_uniform','zero', 'glorot_normal']:
#                        for activation in ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']:
#                            i += 0
#                            print i
#                            params = {"activation" : activation, "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer, "activation_last": "softmax", "optimizer": optimizer, "metrics":["accuracy"], "dropoutRate": dropoutRate}
#                            params["batch_size"] = batch_size
#                            params["epochs"] = epochs
#                            seed = 4444
#                            np.random.seed(seed)
#                            model = SequentialModel(layers, params)
#                            callbacks = DefineCallbacks(modelpath)
#                            model.fit(data_train, labels_train, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1, validation_data=(data_val,labels_val), callbacks=callbacks)
#                            predictions = model.predict(data_test)
#                            fpr, tpr, thr = roc_curve(labels_test[:,sample_names.index("Higgs")], predictions[:,sample_names.index("Higgs")])
#                            print params
#                            par_list.append(params)
#                            auc_list.append(auc(fpr,tpr))
#                            print "Auc:", auc(fpr,tpr)
#
#quit()
#
#

#keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)   #batch normalization, normally is not tuned further
# model.compile(loss='binary_crossentropy', optimizer=adam,metrics=['accuracy', 'fmeasure'])




#
#
# dict_var = {"Info_dict": Info_dict,
#             "isSubset" : True,
#             "isGen" : False,
#             "filePath" : "/beegfs/desy/user/amalara/",
#             "modelpath" : "model_analyze",
#             "varnames" : ["JetInfo", "JetVariables"],
#             "variables" : ["jetPt", "jetEta"],
#             "sample_names": sorted(["Higgs", "QCD", "Top" ]),
#             "radius" : "AK8",
#             "pt_min" : 300,
#             "pt_max" : 500,
#             "max_size" : 10000,
#             "layers" : [3,4,2],
#             "params" : {"epochs" : 12, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
#             "seed" : 4444
#             }
#
#
#
# dict_var = {"Info_dict": Info_dict,
#             "isSubset" : True,
#             "isGen" : False,
#             "filePath" : "/beegfs/desy/user/amalara/",
#             "modelpath" : "model_analyze",
#             "varnames" : ["JetInfo", "JetVariables"],
#             "variables" : ["jetPt", "jetEta", "jetPhi", "jetMass", "jetEnergy", "jetTau1", "jetTau2", "jetTau3", "ncandidates", "jetBtag", "jetTau21", "jetTau31", "jetTau32"],
#             "sample_names": sorted(["Higgs", "QCD", "Top" ]),
#             "radius" : "AK8",
#             "pt_min" : 300,
#             "pt_max" : 500,
#             "max_size" : 10000,
#             "layers" : [50,50,50,50,50,50,10],
#             "params" : {"epochs" : 30, "batch_size" : 512, "activation" : "relu", "kernel_initializer": "glorot_normal", "bias_initializer": "ones", "activation_last": "softmax", "optimizer": "adam", "metrics":["accuracy"], "dropoutRate": 0.01},
#             "seed" : 4444
#             }
#
# np.random.seed(dict_var["seed"])
#
# NN = SequentialNN(dict_var)
# NN.InputShape()
# NN.CreateSubSet()
# NN.Normalization()
# NN.SequentialModel()
# NN.FitModel()
# NN.Predict()
# NN.SaveModel()
#
# def RELU(v):
#     return np.maximum(v, 0)
#
#
# def softmax(v):
#     x = v[0,:]
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
# model = NN.model
# arch = json.loads(open(NN.modelpath+"mymodeljson.json").read())
#
# i_i = 10
# sum = 0
#
# for i_i in range(0,max_):
#     inputs = NN.data_train[i_i:i_i+1,:]
#     outputs = NN.model.predict(NN.data_train[i_i:i_i+1,:])
#     test = NN.data_train[i_i:i_i+1,:]
#     for ind, l in enumerate(model.get_config()):
#         # print l['class_name']
#         if l['class_name'] == "Dropout":
#             continue
#         # print l['class_name']
#         if l['class_name'] == "Dense":
#             # print len(model.layers[ind].get_weights()), model.layers[ind].get_weights()[0].shape, model.layers[ind].get_weights()[1].shape
#             test = np.matmul(test, model.layers[ind].get_weights()[0])
#             test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#             if l["config"]["activation"] == "relu":
#                 test = RELU(test)
#             if l["config"]["activation"] == "softmax":
#                 test = softmax(test)
#         if l['class_name'] == "BatchNormalization":
#             test = test - np.expand_dims(model.layers[ind].get_weights()[2],axis=0)
#             test = test / np.sqrt((np.expand_dims(model.layers[ind].get_weights()[3],axis=0)+0.001))
#             test = test * np.expand_dims(model.layers[ind].get_weights()[0],axis=0)
#             test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#     sum = np.abs(test- outputs) + sum
#
# print sum
#
#
# max_ = 100
# for value in [0,1,2,3]:
#     add = [0,0,0,0]
#     for i_i in range(0,max_):
#         min = 10
#         index = 0
#         inputs = NN.data_train[i_i:i_i+1,:]
#         outputs = NN.model.predict(NN.data_train[i_i:i_i+1,:])
#         test_ = 0
#         for p in multiset_permutations(np.array([0, 1, 2, 3])):
#             test = NN.data_train[i_i:i_i+1,:]
#             for ind, l in enumerate(arch["config"]):
#                 # print l['class_name']
#                 if l['class_name'] == "Dropout":
#                     continue
#                 # print l['class_name']
#                 if l['class_name'] == "Dense":
#                     # print len(model.layers[ind].get_weights()), model.layers[ind].get_weights()[0].shape, model.layers[ind].get_weights()[1].shape
#                     test = np.matmul(test, model.layers[ind].get_weights()[0])
#                     test = test + np.expand_dims(model.layers[ind].get_weights()[1],axis=0)
#                     if l["config"]["activation"] == "relu":
#                         test = RELU(test)
#                     if l["config"]["activation"] == "softmax":
#                         test = softmax(test)
#                 if l['class_name'] == "BatchNormalization":
#                     test = test - np.expand_dims(model.layers[ind].get_weights()[p[0]],axis=0)
#                     test = test / np.sqrt((np.expand_dims(model.layers[ind].get_weights()[p[1]],axis=0)+0.001))
#                     test = test * np.expand_dims(model.layers[ind].get_weights()[p[2]],axis=0)
#                     test = test + np.expand_dims(model.layers[ind].get_weights()[p[3]],axis=0)
#             temp = np.sum(np.abs(test-outputs))
#             if temp < min :
#                 min = temp
#                 index = p
#                 test_ = test
#         print min, index
#         print test_
#         print outputs
#         for t in [0,1,2,3]:
#             if index[t] == value :
#                 add[t] = add[t] + 1./max_
#     print value, add
