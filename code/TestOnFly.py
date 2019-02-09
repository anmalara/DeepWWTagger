from Models import *

path = "/beegfs/desy/user/amalara/output_varariables/Sequential/model_AK8_pt_300_500/model_500epochs_allstat_reduced/"

dict_var = json.loads(open(path+"mymodelinfo.json").read())

np.random.seed(dict_var["seed"])

NN = SequentialNN(dict_var, False)
NN.InputShape()
NN.CreateSubSet()
NN.Normalization()
NN.model = load_model(path+"model_epoch022_loss0.57.h5")
NN.Predict()
NN.Plots(show_figure = False, save_figure = True, extraName="epoch006")

quit()
