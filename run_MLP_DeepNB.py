import os
from Scripts.SrcDeepNB import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

AE_ver = 'v3'
if AE_ver is 'v1':
	from Scripts.SrcAutoEncoder_V1 import * # 
elif AE_ver is 'v2':
	from Scripts.SrcAutoEncoder_V2 import * #		
elif AE_ver is 'v3':
	from Scripts.SrcAutoEncoder_V3 import * # V3 is used currently in the project.		

# Load NeuroBlastoma (NB) dataset	
curr_wd = './DeepNB/'
data_format = "Raw"
data_cache = MLP_data_loading(curr_wd, data_format)

# Parameters for AE_models
drop_rates = [2]
acti_funcs = ['tanh']


# Main program 
for drop_rate_1 in drop_rates:
	for acti_func in acti_funcs:
	
		# AE model settings. 
		randSeed = 2018
		epochs_AE = 1000
		featureOut = 1024
		# rs_param = range(0,50)
		rs_param = np.random.rand(10)*10000
		# rs_param = [11]
		
		drop_rate = drop_rate_1/10
		curr_Dir = (curr_wd + "Models/" + data_format+"_" + acti_func+'_' + 'DROPOUT'+str(drop_rate_1)+'_'+
					str(epochs_AE)+'EPO_' + str(featureOut)+'FEATS_' + AE_ver+"_" + str(randSeed)+"/" 
				   )
				   
		# Endpoint settings.		 
		# PLATFORMS = ['AG','NGS']
		PLATFORMS = ['NGS']
		ENDPOINTS = ['D_FAV_All' , 'B_OS_All' , 'F_OS_HR']
		# modetypes = [['A','C','E']]
		modetypes = [['A'],['C'],['E']]
		
		print("Currently used AE_model: ", curr_Dir)
		print(modetypes)
		# Print Header:
		print("Seed\tPlatform\t"+"\t".join(ENDPOINTS))
		AE_models = load_AEModels(curr_Dir)
		for j, randSeed_MLP in enumerate(rs_param):
			main(AE_models = AE_models, data_cache = data_cache, curr_Dir = curr_Dir,  
				 # Endpoint settings.
				 PLATFORMS = PLATFORMS,
				 ENDPOINTS = ENDPOINTS,
				 modetypes = modetypes,
				 # MLP model settings.
				 randSeed_MLP = int(randSeed_MLP), epochs_MLP = 500, start_epo = 1000, interval = 10, ae_layer = 1,
				 # Other settings.
				 AE_ver = AE_ver, drop_rate_MLP = 0, optimizer = 'adadelta', mode = 'regression',
				 earlyStop = True, cross_val = True, k=5,
				 verbose = 0
				)
	