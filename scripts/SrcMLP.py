# load dependencies:
import numpy as np
from tensorflow.keras.initializers import glorot_uniform, RandomUniform
from tensorflow.keras import regularizers 
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
import random
from math import ceil, sqrt

class IntervalEvaluation(Callback):
	def __init__(self, validation_data=(), interval=10, start_epo = 100, learn_mode = 'regression'):
		super(Callback, self).__init__()

		self.interval = interval
		self.start_epo = start_epo
		self.X_val, self.y_val = validation_data
		self.mode = learn_mode

	def on_epoch_end(self, epoch, logs={}):
		if (epoch % self.interval == 0) and (epoch >= self.start_epo):
			y_val  = self.y_val
			y_pred = self.model.predict(self.X_val, verbose=0)
			# print(y_pred)
			# print(self.mode)
			if self.mode is 'regression':
				y_pred_int=[]
				for pred in y_pred:
					if pred>0.5:
						y_pred_int.append(1)
					else:
						y_pred_int.append(0)
				
				y_val_int=[]          
				for pred in y_val:
					if pred>0.5:
						y_val_int.append(1)
					else:
						y_val_int.append(0)  
			else:
				y_pred_int=[]
				for pred in y_pred:
					if pred[0]>0.5:
						y_pred_int.append(1)
					else:
						y_pred_int.append(0)
				
				y_val_int=[]          
				for pred in y_val:
					if pred[0]>0.5:
						y_val_int.append(1)
					else:
						y_val_int.append(0)  
				
			
			# print(y_pred)
			auc_score = roc_auc_score(y_val, y_pred)
			acc_score = accuracy_score(y_val_int, y_pred_int)			
			mcc_score = matthews_corrcoef(y_val_int, y_pred_int)
			
			# logging.info("interval evaluation - epoch: {:d} - AUC score: {:.6f}".format(epoch, score))
			print("interval evaluation - epoch: {:d}; - fit_loss: {:.3f}; - fit: {:.3f}; - AUC: {:.3f}; - ACC: {:.3f}; - MCC:{:.3f}"
				  .format(epoch, logs['loss'], logs['acc'], auc_score, acc_score, mcc_score))


def MLP_struct(shapeIn, activation = 'relu', drop_rate = 0, depth = 4, seed = None, mode = 'regression'):
	# The dense network model used in this analysis is:
	# n -> 1024 -> 256 -> 128 -> 64 -> out_n (2)
	
	xInput = Input((shapeIn,))
	
	# H1
	net_nodes = 1024
	x = Dense(net_nodes, activation=activation, kernel_initializer=glorot_uniform(0)
			 )(xInput)
	
	for i in range(1,depth):
		net_nodes = int(net_nodes/2)
		x = BatchNormalization()(x)
		# x = Dropout(rate=drop_rate)(x)	
		x = Dense(net_nodes, activation=activation)(x)
		
	# Final BN
	# x = BatchNormalization()(x)   
	
	if mode is 'regression':
		xOutput = Dense(1, activation="sigmoid")(x)
	else: 
		xOutput = Dense(2, activation="softmax")(x)

	# Return a dense model object.
	return Model(xInput, xOutput)

def build_MLP(x, y, xTest = None, yTest = None,
			  # parameters for MLP model structure
			  drop_rate=0, depth = 4, MLP_activation = 'relu',
			  # parameters for MLP model optimizing process
			  mode = 'regression',
			  optimizer = 'adadelta', metrics = ['acc'],
			  # parameters for MLP model training/fitting
			  epochs = 1000, batch_size = 128, earlyStop = False,  
			  # parameters for MLP callbacks
			  start_epo = 100, interval = 5, 
			  # Other parametes 
			  randSeed = None, verbose = 0
			 ):
	if mode is 'regression':
		loss = 'mse'
	else:
		loss = 'categorical_crossentropy'
	
	# Construct MLP Model
	# Reproducibility sets
	shapeIn = x.shape[1]
	np.random.seed(randSeed)
	modelDense = MLP_struct(shapeIn, drop_rate=drop_rate, depth = depth, activation=MLP_activation, seed = randSeed, mode=mode)
	modelDense.compile(loss = loss, optimizer = optimizer, metrics = metrics)
	# Define Callbacks in training process
	if xTest is not None:
		ival = IntervalEvaluation(validation_data=(xTest, yTest), interval=interval, start_epo=start_epo, learn_mode=mode)
		if earlyStop == True:
			es = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, verbose=0, mode='auto')
			callbacks = [es, ival]
		else:
			callbacks = [ival]
	else:
		if earlyStop == True:
			es = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, verbose=0, mode='auto')
			callbacks = [es]
		else:
			callbacks = None

	# Training model 
	np.random.seed(randSeed)
	#print(x[0:5,0:5])
	history = modelDense.fit(x = x, y = y, 
							 epochs = epochs, batch_size = batch_size, 
							 callbacks=callbacks, shuffle=False,
							 verbose=verbose
							)
	
	# Return both trained model and training logs if needed.
	return (modelDense, history)

def save_MLP(Folder, PLATFORM, customStr, model=None, usedEndpoint=''):
	if model is not None:
		modelDenseName = Folder + "/" + PLATFORM + '.' + usedEndpoint + ".MLP"+customStr + ".h5"
		#modelDenseName = "./DeepNB/Models/NoAE/" + PLATFORM + ".B.h5"
		model.save(filepath=modelDenseName)
	else:
		print('No Model Saved.')

def load_MLP(Folder, PLATFORM, customStr, usedEndpoint=''):
	modelDenseName = Folder + "/" + PLATFORM + ".denModel"+customStr + '.' + usedEndpoint+".h5"
	return load_model(modelDenseName)

