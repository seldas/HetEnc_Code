# load dependencies:
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score

def OntHotConverter(y):
	labelEncoder = LabelEncoder()
	yInt = labelEncoder.fit_transform(y)
	oneHotEncoder = OneHotEncoder(sparse=False)
	yInt = yInt.reshape(len(yInt), 1)
	yOneHot = oneHotEncoder.fit_transform(yInt)

	return (y, yInt, yOneHot)
	
def predictEndpoint(modelDense, xTest, yTest=None, mode = 'regression'):
	y_pred = modelDense.predict(xTest, verbose=0)

	if yTest is not None:
		if mode is 'classification':
			y_pred_int=[]
			for pred in y_pred:
				#if pred[0]>0.5:
				if pred[1]>0.5:
					y_pred_int.append(1)
				else:
					y_pred_int.append(0)

			y_val_int=[]		  
			for pred in yTest:
				#if pred[0]>0.5:
				if pred[1]>0.5:
					y_val_int.append(1)
				else:
					y_val_int.append(0)	 
		else:
			y_pred_int=[]
			for pred in y_pred:
				#if pred[0]>0.5:
				if pred>0.5:
					y_pred_int.append(1)
				else:
					y_pred_int.append(0)

			y_val_int=[]		  
			for pred in yTest:
				#if pred[0]>0.5:
				if pred>0.5:
					y_val_int.append(1)
				else:
					y_val_int.append(0)	 

		auc_score = roc_auc_score(yTest, y_pred)
		acc_score = accuracy_score(y_val_int, y_pred_int)
		mcc_score = matthews_corrcoef(y_val_int, y_pred_int)
		
		#print("Prediction performance - ACC: {:.3f}; - AUC: {:.3f}; - MCC: {:.3f}"
		#	  .format(acc_score, auc_score, mcc_score)
		#	 )
	return(auc_score)