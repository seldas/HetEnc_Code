### auto-encoder V3 is a simple version of V1. for which we didn't do afterward-tuning to save computation time.
### Also, we did some other modifications on the auto-encoder network structure.

# load dependencies:
import os
import numpy as np
import sys

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback


##### define auto-encoder model structure #####
def AEModel_struct(shapeIn, shapeOut=256, drop_rate=0, k=4,
				   activationEncoder = "relu", activationDecoder = "sigmoid", BatchNorm = False
				  ):  
	# The auto-encoders used in this study have the same structure:
	# n -> out_n*4 -> out_n*3 -> out_n*2 -> out_n -> out_n*2 -> out_n*3 -> out_n*4 -> n'
	
	# Input
	xInput = Input((shapeIn,))
	x = Dropout(rate=drop_rate)(xInput)
	
	# EN-Layer	
	for i in range(1,k):
		x = Dense(shapeOut*(k+1-i), activation=activationEncoder, name = "en_layer_"+str(i))(x)
		if BatchNorm is True:
			x = BatchNormalization()(x)
		x = Dropout(rate=drop_rate)(x)

	# Mi-Layer (AE Output)
	x = Dense(shapeOut,    activation=activationEncoder, name = "mi_layer")(x)
	if BatchNorm is True:
		x = BatchNormalization()(x)
	x = Dropout(rate=drop_rate)(x)

	# De-Layer
	for j in range(1,k):
		x = Dense(shapeOut*(1+j),    activation=activationEncoder, name = "de_layer_"+str(j))(x)
		if BatchNorm is True:
			x = BatchNormalization()(x)
		x = Dropout(rate=drop_rate)(x)

	# Output
	xOutput = Dense(shapeIn, activation=activationDecoder, name = "recon_sample")(x)  
   
	# return a AutoEncoder Model object.
	return Model(xInput, xOutput)

def build_AEModel_Init(x, y, 
					   featureOut = 256, drop_rate=0.2, acti_func = 'relu',
					   opt = 'adadelta', loss = 'mse', 
					   epochs = 100, batch_size = 128, earlyStop = False, shuffle = True,
					   randSeed = None, verbose = 0
					  ):
	
	np.random.seed(randSeed)
	
	featureIn = x.shape[1]
	
	# No early stop by default
	if earlyStop is True:
		es = EarlyStopping(monitor='loss', min_delta=0.001, patience=50, verbose=0, mode='auto')
	else:
		es = None
	
	aemodel = AEModel_struct(featureIn, featureOut, drop_rate=drop_rate, activationEncoder = acti_func)
	aemodel.compile(loss = loss, optimizer = opt)
	aemodel.fit(x=x, y=y, 
				epochs = epochs, batch_size = batch_size, 
				shuffle=shuffle, callbacks=[es], 
				verbose=verbose
			   )
	
	print("One Auto-Encoder has been initialized!")
	return aemodel

def loadPretrainedModel(modelNames):
	modelAll = []
	for name in modelNames:
		modelTmp = load_model(name)
		modelAll.append(modelTmp)
	
	return modelAll

def measureIntermediateFeature(x, modelAE, layer = 1):
	# layerIntermediate = Model(inputs = modelAE.input, outputs = modelAE.get_layer(name = "recon_sample").output)
	layerIntermediate = Model(inputs = modelAE.input, outputs = modelAE.get_layer(name = "mi_layer").output)
	#layerIntermediate = modelAE
	outputs = layerIntermediate.predict(x)
	
	if layer >= 2:
		layerIntermediate = Model(inputs = modelAE.input, outputs = modelAE.get_layer(name = "en_layer_3").output)
		outputs = np.column_stack((outputs,layerIntermediate.predict(x)))
		
	if layer >= 3:
		layerIntermediate = Model(inputs = modelAE.input, outputs = modelAE.get_layer(name = "en_layer_2").output)
		outputs = np.column_stack((outputs,layerIntermediate.predict(x)))
	
	if layer >= 4:
		layerIntermediate = Model(inputs = modelAE.input, outputs = modelAE.get_layer(name = "en_layer_1").output)
		outputs = np.column_stack((outputs,layerIntermediate.predict(x)))
	
	return outputs	
	
	
def save_AEModels(customStr, models):
	# AE models are shared among endpoints. Since it is unsupervised.
	
	# output file name
	
	(model_A_1_name, model_A_2_name, model_C_name, model_E_name) = AEModelNames(customStr)
	
	(model_A_1, model_A_2, model_C, model_E) = models
	
	model_A_1.save(filepath= model_A_1_name) # AE-Single
	model_A_2.save(filepath= model_A_2_name)
	model_C.save(filepath= model_C_name) # AE-Comb
	model_E.save(filepath= model_E_name) # AE-Cross

	
def load_AEModels(customStr):
	return loadPretrainedModel(AEModelNames(customStr))


def AEModelNames(customStr):
	model_A_1_name= customStr+"AE.Model_A_1.h5"
	model_A_2_name= customStr+"AE.Model_A_2.h5"
	model_C_name  = customStr+"AE.Model_C.h5"
	model_E_name  = customStr+"AE.Model_E.h5"
	
	return (model_A_1_name, model_A_2_name, model_C_name, model_E_name)
	

def train_AEModels(x, xCross, customStr, featureOut = 256, 
				   opt = 'adadelta', loss = 'mse',
				   epochs = 100, batch_size = 128,
				   earlyStop = True, shuffle = True,
				   randSeed = 0, drop_rate=0.2,
				   acti_func= 'relu',
				   verbose = 0
				  ):
	
	xComb = np.row_stack((x,xCross))
	xRev  = np.row_stack((xCross,x))
	
	# AEmodel - Single AE p1
	print('Training AE model for platform A: ')
	model_A_1   = build_AEModel_Init(x=x, y=x, 
									 featureOut = featureOut, drop_rate=drop_rate, acti_func=acti_func,
									 opt = opt, loss = loss, 
									 epochs = epochs, batch_size = batch_size, earlyStop = earlyStop, shuffle = shuffle, 
									 randSeed = randSeed, verbose = verbose
									)
	
	# AEmodel - Single AE p2
	print('Training AE Model for platform B: ')
	model_A_2   = build_AEModel_Init(x=xCross, y=xCross, 
									 featureOut = featureOut, drop_rate=drop_rate, acti_func=acti_func,
									 opt = opt, loss = loss, 
									 epochs = epochs, batch_size = batch_size, earlyStop = earlyStop, shuffle = shuffle, 
									 randSeed = randSeed, verbose = verbose
									)
	
	# AEmodel - CombNet
	print('Training CombNet Model: ')
	model_C   = build_AEModel_Init(x=xComb, y=xComb, 
								   featureOut = featureOut, drop_rate=drop_rate, acti_func=acti_func,
								   opt = opt, loss = loss, 
								   epochs = epochs, batch_size = batch_size, earlyStop = earlyStop, shuffle = shuffle, 
								   randSeed = randSeed, verbose = verbose
								  )
	
	# AEmodel - CrossNet
	print('Training CrossNet Model: ')
	model_E   = build_AEModel_Init(x=xComb, y=xRev, 
								   featureOut = featureOut, drop_rate=drop_rate, acti_func=acti_func,
								   opt = opt, loss = loss, 
								   epochs = epochs, batch_size = batch_size, earlyStop = earlyStop, shuffle = shuffle, 
								   randSeed = randSeed, verbose = verbose
								  )
	
	models = [model_A_1, model_A_2, model_C, model_E]
	
	save_AEModels(customStr, models)
	return models
	
def ConvertFeat(x, AE_models, modetype = ['A'], layer = 1):
	(model_A, model_C, model_E)=AE_models
	xFeats = None

	for mt in modetype:    
		if mt is 'A':
			xFeat = measureIntermediateFeature(x, model_A, layer = layer)
		elif mt is 'C':
			xFeat = measureIntermediateFeature(x, model_C, layer = layer)
		elif mt is 'E':
			xFeat = measureIntermediateFeature(x, model_E, layer = layer)
		elif mt is 'NoAE':
			xFeat = x
		elif mt is '' :
			xFeat = x
		else:
			xFeat = None
			print("Not identified converter!")
			sys.exit()

		# Stack different type of features 
		if xFeats is not None:    
			xFeats = np.column_stack((xFeats, xFeat))
		else:
			xFeats = xFeat # Initialize at the first time.

	return(xFeats)