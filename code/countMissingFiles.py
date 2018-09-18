from variables import *




path = "/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/input_varariables/NTuples_Tagger/"


@timeit
def countselectBranches(path="/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/input_varariables/NTuples_Tagger/"):
    for bkg in bkgs:
        for radius in radii:
            max_ = 0
            for var in ["gen_cand", "gen_jet", "cand", "jet"]:
                with open(path+bkg+"_"+radius+"/"+var+"_var_missing.txt", "w") as file:
                    count = 0
                    total = 0
                    for i in range(files_dictionary[bkg][0]+1):
                        if str(i) in files_dictionary[bkg][1]:
                            continue
                        total += 1
                        file_name = path+bkg+"_"+radius+"/"+var+"_var_"+bkg+"_"+str(i)+".npy"
                        if not os.path.isfile(file_name):
                            file.write(str(i)+"\n")
                            count += 1
                    file.write("Total Missing: "+str(count)+"/"+str(total))



@timeit
def countConv2D_variables(path="/nfs/dust/cms/user/amalara/WorkingArea/File/NeuralNetwork/input_varariables/NTuples_Tagger/"):
    for bkg in bkgs:
        for radius in radii:
            max_ = 0
            for pt in ["300_500", "500_10000"]:
                with open(path+"JetImage/"+bkg+"_"+radius+"/JetImage_matrix_"+bkg+"_"+radius+"_pt_"+pt+"_missing.txt", "w") as file:
                    count = 0
                    total = 0
                    for i in range(files_dictionary[bkg][0]+1):
                        if str(i) in files_dictionary[bkg][1]:
                            continue
                        total += 1
                        file_name = path+"JetImage/"+bkg+"_"+radius+"/JetImage_matrix_"+bkg+"_"+radius+"_file_"+str(i)+"_pt_"+pt+".npy"
                        if not os.path.isfile(file_name):
                            file.write(str(i)+"\n")
                            count += 1
                    file.write("Total Missing: "+str(count)+"/"+str(total))

countselectBranches()
countConv2D_variables()
