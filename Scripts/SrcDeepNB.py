# load dependencies:
import numpy as np
import pandas as pd
import os
import random

from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

from sklearn.preprocessing import StandardScaler as Norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, accuracy_score

from Scripts.SrcMLP import * 
from Scripts.SrcDataPrep import *

AE_ver = 'v3'
if AE_ver is 'v1':
	from Scripts.SrcAutoEncoder_V1 import * # 
elif AE_ver is 'v2':
	from Scripts.SrcAutoEncoder_V2 import * #		
elif AE_ver is 'v3':
	from Scripts.SrcAutoEncoder_V3 import * # V3 is used currently in the project.			

def MLP_data_loading(curr_wd, data_format):
	data_wd = curr_wd + data_format + '/'
	
	data_AG = pd.read_csv(data_wd+'NB_AG_121.txt',delimiter="\t") 
	data_NGS = pd.read_csv(data_wd+'NB_NGS_121.txt',delimiter="\t")
	clin_info = pd.read_csv(curr_wd+'SEQC_NB_249_ValidationSamples_ClinicalInfo_20121128.txt', delimiter="\t")
	
	TrainingID = clin_info.loc[0::2,'SEQC_NB_SampleID']
	TestingID  = clin_info.loc[1::2,'SEQC_NB_SampleID']

	# Assign Training/Testing dataset based on SEQC group
	xTrain_1 = np.transpose(data_AG.loc[:,TrainingID.values.tolist()].values)
	xTrain_2 = np.transpose(data_NGS.loc[:,TrainingID.values.tolist()].values)
	
	xTest_1 = np.transpose(data_AG.loc[:,TestingID.values.tolist()].values)
	xTest_2 = np.transpose(data_NGS.loc[:,TestingID.values.tolist()].values)
		
	data_cache = (xTrain_1, xTest_1, xTrain_2, xTest_2, clin_info)
	return data_cache

def main(AE_models, data_cache, curr_Dir, 
		 # Endpoint settings.
		 PLATFORMS = ['AG','NGS'],
		 ENDPOINTS = ['D_FAV_All' , 'A_EFS_All' , 'E_EFS_HR'],
		 modetypes = [['A','C','E']],
		 # MLP model settings.
		 randSeed_MLP = 2018, epochs_MLP = 1000, start_epo = 100, interval = 10, ae_layer = 1,
		 # Other settings.
		 AE_ver = 'v3', drop_rate_MLP = 0,	optimizer = 'adadelta', mode = 'regression',
		 earlyStop = True, cross_val = False, k=5,
		 verbose = 0
		):
	
	### Load dataset ###
	(xTrain_1, xTest_1, xTrain_2, xTest_2, clinical_info) = data_cache
	####################
	
	for modetype in modetypes: 
		for PLATFORM in PLATFORMS:
			customStr_2 = '.RS'+str(randSeed_MLP)+'.'+str(epochs_MLP)+'EPO.Layer'+str(ae_layer)+'.Model_'+"&".join(modetype)
			print("Current Strategy:", modetype ,'Platform: ', PLATFORM)
			
			if PLATFORM is 'AG':
				xTrainPL, xTestPL = (xTrain_1, xTest_1)
			else:
				xTrainPL, xTestPL = (xTrain_2, xTest_2)		
						
			if cross_val is True:
			# Cross validation
				cache = (xTrainPL, clinical_info, curr_Dir, customStr_2)
				curr_AUCs = cv_MLP(PLATFORM, ENDPOINTS, cache, k=k,
								   AE_models=AE_models, modetype = modetype, ae_layer = ae_layer, drop_rate = drop_rate_MLP,
								   optimizer = optimizer, mode=mode, AE_ver = AE_ver,
								   epochs=epochs_MLP, start_epo = start_epo, interval = interval, earlyStop = earlyStop, 
								   randSeed = randSeed_MLP, verbose = verbose
								  )
				print("Seed: "+str(randSeed_MLP) + "\tCV\t"+ PLATFORM + "\t" + curr_AUCs)
			else:
			# External Test
				cache = (xTrainPL, xTestPL, clinical_info, curr_Dir, customStr_2)
				curr_AUCs = train_MLP(PLATFORM, ENDPOINTS, cache, 
									  AE_models=AE_models, modetype = modetype, ae_layer = ae_layer, drop_rate = drop_rate_MLP,
									  optimizer = optimizer, mode=mode, AE_ver = AE_ver,
									  epochs=epochs_MLP, start_epo = start_epo, interval = interval, earlyStop = earlyStop, 
									  #Den_exists = True, 
									  Den_save = False, 
									  randSeed = randSeed_MLP, verbose = verbose
									 )
				print("Seed "+str(randSeed_MLP) + "  \tTest\t" + PLATFORM + "\t" + curr_AUCs)		 

															 
def preparationY(usedEndpoint, clinical_info, x, xTest, xCross=None, xTestCross=None):
	# Prepare The NeuroBlastoma dataset based on the availability of endpoint.
	yTrain = clinical_info.loc[0::2 , usedEndpoint].values
	yTest  = clinical_info.loc[1::2 , usedEndpoint].values

	# Remove label contains NaN.
	validSampleTrain = np.logical_not(np.isnan(yTrain))
	validSampleTest = np.logical_not(np.isnan(yTest))
	
	yTrain = yTrain[np.logical_not(np.isnan(yTrain))]
	yTest = yTest[np.logical_not(np.isnan(yTest))]
	
	# Convert Label into OneHot Format.
	yTrain, yTrainInt, yTrainOneHot = OntHotConverter(yTrain)
	yTest, yTestInt, yTestOneHot = OntHotConverter(yTest)

	# Remove Samples with NaN Label
	x          = x[validSampleTrain,:]
	xTest      = xTest[validSampleTest,:]
	
	if xCross is not None:
		xCross     = xCross[validSampleTrain,:] 
		xTestCross = xTestCross[validSampleTest,:]
		return (x, xTest, xCross, xTestCross, yTrainOneHot, yTestOneHot, yTrain, yTest)
	else:
		return (x, xTest, yTrainOneHot, yTestOneHot, yTrain, yTest)  
	
def cv_MLP(PLATFORM, ENDPOINTS, cache, k=5,
			  # parameters for AE converter
			  AE_models=None, modetype = "A", ae_layer = 1, AE_ver = 'v3',
			  # parameters for MLP training
			  epochs=100, batch_size = 128, earlyStop = False,
			  mode = 'regression', drop_rate = 0,
			  optimizer = 'adadelta',
			  # parameters for Callbacks
			  start_epo = 100, interval = 1,
			  randSeed = 0, verbose = 0
			 ):
	
	curr_AUCs = None
	(xTrainPL, clinical_info, currWorkingDirectory, customStr_2) = cache
	
	for usedEndpoint in ENDPOINTS: 
		## Step 3: Predictive Modeling
		(xTrain, yTrain, folds) = generateKFold(usedEndpoint, clinical_info, x=xTrainPL, k=k, randSeed=randSeed, mode = mode)
		#print(folds)
		if AE_models is not None:
			used_model = select_model(PLATFORM, AE_models, AE_ver)
			xTrainFeat = ConvertFeat(xTrain, used_model, modetype, layer = ae_layer)				
		else:
			print('Warning: AE_model is not defined. Use Raw Features.')
			xTrainFeat = xTrain
		
		# print("Primary Dataset (Input):", PLATFORM, "; Endpoint: ", usedEndpoint, ", Feat: ",xTrainFeat.shape)
		# print(xTrainFeat.shape, yTrainOneHot.shape, xTestFeat.shape, yTestOneHot.shape)
		
		y_pred_total = None
		y_val_total  = None
		for j, (train_idx, val_idx) in enumerate(folds):
			# print(j, val_idx)
			x_train_cv = xTrainFeat[train_idx]
			y_train_cv = yTrain[train_idx]
			x_valid_cv = xTrainFeat[val_idx]
			y_valid_cv = yTrain[val_idx]
			#print(y_valid_cv.shape)
			
			# Normalization
			scaler_1 = Norm()
			scaler_1.fit(x_train_cv)
			x_train_cv = scaler_1.transform(x_train_cv)
			x_valid_cv  = scaler_1.transform(x_valid_cv)
						
			(modelDense, history) = build_MLP(x = x_train_cv, 
											  y = y_train_cv,
											  xTest = x_valid_cv,
											  yTest = y_valid_cv,
											  drop_rate = drop_rate, depth = 4,
											  optimizer = "adadelta",
											  mode =  mode, 
											  MLP_activation = 'relu',
											  epochs = epochs, batch_size = batch_size, earlyStop = earlyStop,  
											  start_epo = start_epo, interval = interval,
											  randSeed = randSeed, verbose = verbose
											 )
			y_pred = modelDense.predict(x_valid_cv)
			y_val = y_valid_cv
			# print(y_pred.shape)
			if y_pred_total is not None:
				y_pred_total = np.concatenate((y_pred_total,y_pred),axis=0)
				y_val_total  = np.concatenate((y_val_total,y_val),axis=0)
				# print(y_val_total.shape)
			else:
				y_pred_total = y_pred
				y_val_total  = y_val
				
			del(modelDense, history)

		y_pred_int=[]
		for pred in y_pred_total:
			#if pred[0]>0.5:
			if pred>0.5:
				y_pred_int.append(1)
			else:
				y_pred_int.append(0)

		y_val_int=[]
		for pred in y_val_total:
			#if pred[0]>0.5:
			if pred>0.5:
				y_val_int.append(1)
			else:
				y_val_int.append(0)	 	
		# print(len(y_val_int), len(y_pred_total))
		cm = confusion_matrix(y_val_int, y_pred_int)
		# print("TP: ",cm[1,1], "TN: ", cm[0,0], "FP: ", cm[0,1], "FN: ", cm[1,0])
		
		acc = accuracy_score(y_val_int, y_pred_int)
		auc = roc_auc_score(y_val_total, y_pred_total)
		mcc = matthews_corrcoef(y_val_int, y_pred_int)
		# print("Overall - ACC: {:.3f} - AUC: {:.3f}  - MCC: {:.3f}".format(acc, auc, mcc))
		if curr_AUCs is None:
			curr_AUCs = str("{:.3f}".format(auc))
		else:
			curr_AUCs = curr_AUCs+"\t"+str("{:.3f}".format(auc))
	
	return curr_AUCs

def generateKFold(usedEndpoint, clinical_info, x, 
				  k=5, randSeed=2018, mode = 'regression'
				 ):
	# Prepare The NeuroBlastoma dataset based on the availability of endpoint.
	yTrain = clinical_info.loc[0::2 , usedEndpoint].values
	validSampleTrain = np.logical_not(np.isnan(yTrain))
	yTrain = yTrain[np.logical_not(np.isnan(yTrain))]
	yTrain, yTrainInt, yTrainOneHot = OntHotConverter(yTrain)
	
	# Remove Samples with NaN Label
	x = x[validSampleTrain,:]
	
	folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=randSeed).split(x, yTrain))		   
	if mode is 'regression':
		return (x, yTrain, folds)
	else:
		return (x, yTrainOneHot, folds)

def select_model(PLATFORM, AE_models, AE_ver):
	# print(PLATFORM)
	used_model = None
	if PLATFORM is 'AG':
		if AE_ver is 'v1':
			(model_A_1, model_A_2, model_B_1, model_B_2, model_C, model_D_1, model_D_2, model_E, model_F_1, model_F_2) = AE_models
			used_model = (model_A_1, model_B_1, model_C, model_D_1, model_E, model_F_1)
		elif AE_ver is 'v2':
			(model_A_1, model_A_2, model_B_1, model_B_2, model_C_1, model_C_2) = AE_models			
			used_model = (model_A_1, model_B_1, model_C_1)
		elif AE_ver is 'v3':
			(model_A_1, model_A_2, model_C, model_E) = AE_models
			used_model = (model_A_1, model_C, model_E)
	elif PLATFORM is 'NGS':
		if AE_ver is 'v1':
			(model_A_1, model_A_2, model_B_1, model_B_2, model_C, model_D_1, model_D_2, model_E, model_F_1, model_F_2) = AE_models
			used_model = (model_A_2, model_B_2, model_C, model_D_2, model_E, model_F_2)
		elif AE_ver is 'v2':
			(model_A_1, model_A_2, model_B_1, model_B_2, model_C_1, model_C_2) = AE_models			
			used_model = (model_A_2, model_B_2, model_C_2)
		elif AE_ver is 'v3':
			(model_A_1, model_A_2, model_C, model_E) = AE_models
			used_model = (model_A_2, model_C, model_E)
	
	return used_model

def train_MLP(PLATFORM, ENDPOINTS, cache, 
			  # parameters for AE converter
			  AE_models=None, modetype = "A", ae_layer = 1, AE_ver = 'v3',
			  # parameters for MLP training
			  epochs=100, batch_size = 128, earlyStop = False, 
			  mode = 'regression', drop_rate = 0,
			  optimizer = 'adadelta',
			  # parameters for Callbacks
			  start_epo = 100, interval = 1,
			  # Other parameters
			  Den_exists = False, 
			  Den_save = True, 
			  randSeed = 0, verbose = 0
			 ):
	
	curr_AUCs = None
	(xTrainPL, xTestPL, clinical_info, currWorkingDirectory, customStr_2) = cache
		
	for usedEndpoint in ENDPOINTS: 
		## Step 3: Predictive Modeling
		(xTrain, xTest, 
		 yTrainOneHot, yTestOneHot, 
		 yTrain, yTest) = preparationY(usedEndpoint, clinical_info, 
									   x=xTrainPL, xTest=xTestPL
									  )
		
		if AE_models is not None:
			used_model = select_model(PLATFORM, AE_models, AE_ver)
			xTrainFeat = ConvertFeat(xTrain, used_model, modetype, layer = ae_layer)
			xTestFeat  = ConvertFeat(xTest,  used_model, modetype, layer = ae_layer)		
		else:
			# print('Warning: AE_model is not defined. Use Raw Features.')
			xTrainFeat = xTrain
			xTestFeat  = xTest
	
		# Normalization
			scaler_1 = Norm()
			scaler_1.fit(xTrainFeat)
			xTrainFeat = scaler_1.transform(xTrainFeat)
			xTestFeat  = scaler_1.transform(xTestFeat)
		
		#For Categorical Analysis
		if mode is 'classification':
			yTrain = yTrainOneHot
			yTest  = yTestOneHot
		
		# print("Endpoint: ", usedEndpoint, ", Feat: ",xTrainFeat.shape)
		if Den_exists is False:
			(modelDense, history) = build_MLP(x = xTrainFeat, 
											  y = yTrain, 
											  xTest = xTestFeat, 
											  yTest = yTest, 
											  ### parameters for MLP model structure ###
											  drop_rate = drop_rate, depth = 4,
											  ### parameters for MLP model optimizing process ###
											  optimizer = "adadelta", 
											  mode = mode,
											  MLP_activation = 'relu',
											  ### parameters for MLP model training/fitting ###
											  epochs = epochs, batch_size = batch_size, earlyStop = earlyStop,  
											  ### parameters for MLP callbacks ###
											  start_epo = start_epo, interval = interval,
											  ### Other parametes ### 
											  randSeed = randSeed, verbose = verbose
											 )
			curr_AUC = predictEndpoint(modelDense, xTestFeat, yTest, mode = mode)
			if curr_AUCs is None:
				curr_AUCs = str("{:.3f}".format(curr_AUC))
			else:
				#print(curr_AUCs, curr_AUC)
				curr_AUCs = curr_AUCs+"\t"+str("{:.3f}".format(curr_AUC))
			# print(history)
			if Den_save is True:
			 	save_MLP(currWorkingDirectory, PLATFORM, customStr_2, modelDense, usedEndpoint)
				
			del(modelDense, history)	
		## For existed model evaluation:
		else: 
			(modelDense) = load_MLP(currWorkingDirectory, PLATFORM, customStr_2, usedEndpoint)
			y_pred_train = predictEndpoint(modelDense, xTrainFeat, yTrain, mode = mode)
			y_pred_test  = predictEndpoint(modelDense, xTestFeat,  yTest,  mode = mode)
			
			del(modelDense)
	
	return curr_AUCs