# DeepWWTagger

Steps

1) launch\_selectBranches.sh to convert root to npy format
2) launch\_preProcessing\_\* to prepare inputs for each kind of network:
	Repeat this step 2 times. First with flag\_merge=0 and then with flag\_merge=1. (check inside the code)
	This has been done to work in parallel and speed up the process.
3) launch\_Lola.sh, launch\_Sequential.sh, launch\_JetImage.sh to train networks
