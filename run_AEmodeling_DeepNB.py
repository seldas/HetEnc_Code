# load dependencies:
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler as Norm

from .SrcAutoEncoder_V3 import * 


def main(curr_wd = './Data/', data_format = 'Norm'):		
	#### Load dataset #####
	(x_1, x_2) = AE_data_loading(curr_wd, data_format)
	#######################
	
	###################################
	##### Parameter Setting ###########
	batch_size = x_1.shape[0]
	verbose = 2
	randSeed= 2018
	epochs  = 1000
	featureOuts = [1024]
	acti_funcs = ['tanh']
	drop_rates = [2]
	###################################
	###################################
	
	param_cache = (curr_wd, batch_size, verbose, randSeed, epochs, featureOuts, acti_funcs, drop_rates, data_format)
	AE_modeling(x_1, x_2, param_cache)


#######################################
	
	
def AE_data_loading(curr_wd, data_format):
	### Load dataset
	data_wd = curr_wd + data_format + '/'
	data_AG   = pd.read_csv(data_wd+'NB_AG_121.txt',delimiter="\t") 
	data_NGS  = pd.read_csv(data_wd+'NB_NGS_121.txt',delimiter="\t")
	clin_info = pd.read_csv(curr_wd+'SEQC_NB_249_ValidationSamples_ClinicalInfo_20121128.txt', delimiter ="\t")
	TrainingID = clin_info.loc[0::2,'SEQC_NB_SampleID']

	# Assign Training/Testing dataset based on SEQC group
	x_1 = np.transpose(data_AG.loc[:,TrainingID.values.tolist()].values)
	x_2 = np.transpose(data_NGS.loc[:,TrainingID.values.tolist()].values)

	return (x_1, x_2)

def AE_modeling(x_1, x_2, param_cache):
	(curr_wd, batch_size, verbose, randSeed, epochs, featureOuts, acti_funcs, drop_rates, data_format) = param_cache
	for featureOut in featureOuts:
		for drop_rate_1 in drop_rates:
			for acti_func in acti_funcs:
				print('features: ', featureOut,' drop-out: ', drop_rate_1,' activate function: ', acti_func)
				drop_rate = drop_rate_1/10
				curr_Dir = (curr_wd + "Models/"+ data_format + '_' + acti_func+'_'+
							 'DROPOUT'+str(drop_rate_1)+'_'+
							 str(epochs)+'EPO_'+ str(featureOut)+'FEATS_'+ AE_ver+"_" +str(randSeed)+ "/")

				try:
					os.stat(curr_Dir)
				except:
					os.mkdir(curr_Dir)
					print('Start Running ', curr_Dir, '.')

				AE_models = train_AEModels(x_1, x_2, curr_Dir, drop_rate = drop_rate,
					 randSeed = randSeed, epochs=epochs, batch_size = batch_size, featureOut = featureOut, acti_func=acti_func,
					 verbose = verbose)

######################################

if __name__ == '__main__':
	main()