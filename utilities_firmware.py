import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import time
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from collections import Counter


def load_traces(num_columns: int = 24, num_datasets: int = 1): # num_columns is in bytes
	assert isinstance(num_columns, int), "num_columns must be an integer"
	# Main folder with all traces
	ppath = 'F:\\NBox\MNA Research\Singtel Firmware\Singtel Firmware Data\Traces'
	# Individual Problems
	folder_names = ['AES128', 'Interrupt', 'LED', 'Random', 'Shake', 'Temperature', 'Vibration', 'XTS']
	# Type Names - Genuine etc
	type_names = ['safe', 'mal1', 'mal2', 'mal3']
	full_struct = {} # Should contain all data from all folders included in folder_names
	# for ii, fname in enumerate(folder_names):
	num_datasets = len(folder_names) if num_datasets == -1 else num_datasets
	for ii in range(num_datasets):
		fname = folder_names[ii]
		fold_struct = {}
		for jj, tname in enumerate(type_names):
			file_path = os.path.join(ppath, f'{fname}_{tname}')
			temp_data = load_myfiles(fp=file_path, nc=num_columns)
			fold_struct[tname] = temp_data
		full_struct[fname] = fold_struct
	return full_struct

def load_myfiles(fp=None, nc=24):
	file_list = os.listdir(fp)
	# Filter the list to include only files (not directories)
	file_list = [file for file in file_list if os.path.isfile(os.path.join(fp, file))]
	num_files = len(file_list)
	for ii, file in enumerate(file_list):
		file_path = os.path.join(fp, file)
		
		with open(file_path, 'r') as binary_file:
			hex_text = binary_file.read().strip()
		# Remove newline characters ('\n') from hex_text
		hex_text = hex_text.replace('\n', '')
		hex_text = hex_text.replace('\r', '')
		
		len_text = len(hex_text)
		n_rows = np.ceil(len_text / (nc * 2))
		extra_elems = round((n_rows - (len_text / (nc * 2))) * nc * 2)
		
		hex_text_new = hex_text + '0' * extra_elems
		
		# Initialize an empty list to store bytes
		byte_list = []
		
		# Iterate through the string by grouping two characters at a time
		for i in range(0, len(hex_text_new), 2):
			byte_str = hex_text_new[i:i + 2]  # Get two characters at a time
			byte_value = int(byte_str, 16)  # Convert the hexadecimal string to an integer
			byte_list.append(byte_value)
		
		hex_array = np.array(byte_list, dtype=np.uint8)
		hex_2D = hex_array.reshape(-1, nc)
		out_array = np.expand_dims(hex_2D, 0) if ii == 0 else np.concatenate((out_array, np.expand_dims(hex_2D, 0)),
																			 axis=0)
	# Number of traces x nrows x ncolumns
	return out_array

def extract_data(full_struct):
	out_struct = {}
	for key in full_struct.keys():
		out_prob = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp = inner_struct[inkey]
			data = temp if inkey == 'safe' else np.concatenate((data, temp), axis=0)
			if inkey == 'safe':
				label = np.zeros(shape=(temp.shape[0], 1), dtype=int)
			else:
				label = np.concatenate((label, ii*np.ones(shape=(temp.shape[0], 1), dtype=int)), axis=0)
		out_prob['data'] = data
		out_prob['label'] = label
		out_struct[key] = out_prob
	return out_struct

def struct_normalize(full_struct):
	for key in full_struct.keys():
		out_prob = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			if np.max(inner_struct[inkey].reshape(-1) > 1.1):
				inner_struct[inkey] = inner_struct[inkey] / 255.0

def comp_svd(data, k=1):
	sh = data.shape
	data_res = np.reshape(data, (sh[0], -1))
	U, S, V = np.linalg.svd(data_res.T, full_matrices=False)
	if k == 1:
		temp = ((U[:, 0].reshape(-1, 1) * S[0]) @ V[0, :].reshape(1, -1)).T
	else:
		temp = (U[:, :2] @ np.diag(S[0:2]) @ V[:2, :]).T
	
	recon = data_res - temp
	return recon.reshape(sh), V[:k, :].T

def comp_orthproj(data, method='svd', k=1):
	# k components to be used for orthogonal projection computation
	if method == 'svd':
		U, S, Vh = np.linalg.svd(data, full_matrices=False)
		vv = Vh[:k, :].T if k > 1 else np.expand_dims(Vh[0, :], 1)
	else:   # Power iteration similar to that of dictionary learning S1
		uu = normalize(np.random.randn(data.shape[0], 1), axis=0)
		vv = normalize(np.random.randn(data.shape[1], 1), axis=0)
		
		for ii in range(5): # first left and right singular vectors
			vv = normalize(np.dot(data.T, uu), axis=0)
			uu = normalize(np.dot(data, vv), axis=0)
	orthProj = np.eye(data.shape[-1]) - np.dot(vv, vv.T)
	return orthProj
	
def remove_comp(full_struct, method='pow', kcomp=1):
	# we will compute the orhtogonal projection of first singular vector V of safe dataset, and remove its effects
	# from safe as well as unsafe datasets of its respective dataset
	# method using svd or pow
	out_struct = {}
	for key in full_struct.keys():
		inner = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp = inner_struct[inkey].reshape((inner_struct[inkey].shape[0], -1))
			# data = temp if inkey == 'safe' else np.concatenate((data, temp), axis=0)
			if inkey == 'safe':
				orthProj = comp_orthproj(temp, method=method, k=kcomp)
				inner[inkey] = np.dot(temp, orthProj)
			else:
				inner[inkey] = np.dot(temp, orthProj)
		out_struct[key] = inner
	return out_struct

# Keep specific components and remove the rest
def keep_comp(full_struct, method='svd', kcomp=None, scale=False):
	if kcomp is None:
		kcomp = np.arange(1, 10)
	out_struct = {}
	for key in full_struct.keys():
		inner = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp = inner_struct[inkey].reshape((inner_struct[inkey].shape[0], -1))
			# data = temp if inkey == 'safe' else np.concatenate((data, temp), axis=0)
			if inkey == 'safe':
				if scale:
					scaler = StandardScaler()
					temp = scaler.fit_transform(temp)
				U, S, Vh = np.linalg.svd(temp, full_matrices=False)
				V_proj = Vh[kcomp, :].T
				inner[inkey] = np.dot(temp, V_proj)
			else:
				if scale:
					temp = scaler.transform(temp)
				inner[inkey] = np.dot(temp, V_proj)
		out_struct[key] = inner
	return out_struct

# HOG descriptor based pre-processing for feature extraction
def hog_prep(full_struct, kcomp=10):
	from skimage.feature import hog
	
	out_struct = {}
	for key in full_struct.keys():
		inner = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp_struct = inner_struct[inkey]
			for jj in range(temp_struct.shape[0]):
				fd, temp = hog(temp_struct[jj, :, :], orientations=1, pixels_per_cell=(3, 3), cells_per_block=(8, 8), visualize=True,
				               feature_vector=True)
				out_str = fd if jj == 0 else np.vstack([out_str, fd])
			inner[inkey] = out_str
		out_struct[key] = inner
	return out_struct

# Remove the first and keep the rest depending on percentage of
def keep_percentage(full_struct, method='svd', varkeep=None, scale=False):
	if varkeep is None:
		percent = 0.95
	out_struct = {}
	for key in full_struct.keys():
		inner = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp = inner_struct[inkey].reshape((inner_struct[inkey].shape[0], -1))
			# data = temp if inkey == 'safe' else np.concatenate((data, temp), axis=0)
			if inkey == 'safe':
				if scale:
					scaler = StandardScaler()
					temp = scaler.fit_transform(temp)
				U, S, Vh = np.linalg.svd(temp, full_matrices=False)
				cumvar = np.cumsum(S)/np.sum(S)
				kcomp = np.argmax(cumvar >= varkeep) + 1
				V_proj = Vh[1:kcomp, :].T
				inner[inkey] = np.dot(temp, V_proj)
			else:
				if scale:
					temp = scaler.transform(temp)
				inner[inkey] = np.dot(temp, V_proj)
		out_struct[key] = inner
	return out_struct

def pca_reduction(full_struct, kcomp=10):
	out_struct = {}
	for key in full_struct.keys():
		inner = {}
		inner_struct = full_struct[key]
		inner_keys = list(inner_struct.keys())
		for ii, inkey in enumerate(inner_keys):
			temp = inner_struct[inkey].reshape((inner_struct[inkey].shape[0], -1))
			# data = temp if inkey == 'safe' else np.concatenate((data, temp), axis=0)
			if inkey == 'safe':
				pca = PCA(n_components=kcomp)
				inner[inkey] = pca.fit_transform(temp)
			else:
				inner[inkey] = pca.transform(temp)
		out_struct[key] = inner
	return out_struct

# Leave one out test on all datasets
def leaveOneOut(full_struct):
	names = list(full_struct.keys())
	type_names = ['safe', 'mal1', 'mal2', 'mal3']
	# output dataset x numClassifier x type x Metrics
	Results = np.zeros((len(names), 5, 3, 4))
	for ii, key in enumerate(names):
		DS = full_struct[key]
		D_safe = DS[type_names[0]]
		D_mal = [DS[type_names[1]], DS[type_names[2]], DS[type_names[3]]]
		for jj, XX in enumerate(D_mal):
			X_data = np.concatenate([D_safe] + D_mal[:jj] + D_mal[jj+1:], axis=0)
			y_data = np.concatenate((np.zeros((D_safe.shape[0], 1)), np.ones((X_data.shape[0]-D_safe.shape[0], 1))), axis=0).reshape(-1).astype(np.int16)
			y_test = np.ones((XX.shape[0], ))
			temp = all_but_one(X_data, y_data, XX, y_test, model='RFC')
			Results[ii, 0, jj, :] = np.array(temp)
			temp = all_but_one(X_data, y_data, XX, y_test, model='SVM')
			Results[ii, 1, jj, :] = np.array(temp)
			temp = all_but_one(X_data, y_data, XX, y_test, model='KNN')
			Results[ii, 2, jj, :] = np.array(temp)
			temp = all_but_one(X_data, y_data, XX, y_test, model='DT')
			Results[ii, 3, jj, :] = np.array(temp)
			temp = all_but_one(X_data, y_data, XX, y_test, model='GB')
			Results[ii, 4, jj, :] = np.array(temp)
	
	return Results

def all_but_one(X_data, y_data, X_other, y_other, model='RFC'):
	data_shape = X_data.shape
	if len(data_shape) > 2:
		nCols = data_shape[1] * data_shape[2]
		X_data = X_data.reshape((-1, nCols))
		X_other = X_other.reshape((-1, nCols))
	
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=98,
														shuffle=True, stratify=y_data)
	scaler = MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	X_other = scaler.transform(X_other)
	
	if model.lower() == 'rfc':
		model = random_forest_classifier
	elif model.lower() == 'svm':
		model = svm_classifier
	elif model.lower() == 'knn':
		model = knn_classifier
	elif model.lower() == 'dt':
		model = decision_tree_classifier
	elif model.lower() == 'gb':
		model = gradient_boosting_classifier
	else:
		print('Wrong model selected, exiting')
		return 0

	model_trained = model(X_train, y_train)
	print("Results using " + model.__name__ + " On Train-Test dataset")
	scores = perf_rates(y_test, X_test, model=model_trained)
	print("Now testing on unseen dataset: Probabilities")
	scores = perf_rates(y_other, X_other, model=model_trained)
	
	return scores

# True - False Positive and Detection Rate computation
def perf_rates(label, input, model=None):  # if model is given, provide data in prediction var
	prediction = input if model is None else model.predict(input)
	# if predicting a single class, then we need to plug in a label of opposite class in case of 100% accuracy
	label = np.append(label, [0, 1])
	prediction = np.append(prediction, [0, 1])
	
	CFM = metrics.confusion_matrix(label, prediction)
	print(CFM)
	(TN, FP, FN, TP) = CFM.reshape(-1)
	PrD = 100 * TP / (TP + FN)
	PrFA = 100 * FP / (TN + FP)
	PrM = 100 * FN / (FN + TP)
	Acc = metrics.accuracy_score(label, prediction) * 100.0
	print("Accuracy: {:.4f}".format(Acc))
	print("Prob Detection: {:.4f}".format(PrD))
	print("Prob FA: {:.4f}".format(PrFA))
	print("Prob Miss: {:.4f}".format(PrM))
	return Acc, PrD, PrFA, PrM

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% A few classical classifiers
def logistic_reg_classifier(feature, label):
	from sklearn.linear_model import LogisticRegression
	model = LogisticRegression(penalty='l2')
	model.fit(feature, label)
	return model

def random_forest_classifier(feature, label):
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(n_estimators=8)
	model.fit(feature, label)
	return model

def knn_classifier(feature, label):
	from sklearn.neighbors import KNeighborsClassifier
	model = KNeighborsClassifier()
	model.fit(feature, label)
	return model

def decision_tree_classifier(feature, label):
	from sklearn import tree
	model = tree.DecisionTreeClassifier()
	model.fit(feature, label)
	return model

def gradient_boosting_classifier(feature, label):
	from sklearn.ensemble import GradientBoostingClassifier
	model = GradientBoostingClassifier(n_estimators=20)
	model.fit(feature, label)
	return model

def svm_classifier(feature, label):
	from sklearn.svm import SVC
	model = SVC(kernel='rbf', probability=True)
	model.fit(feature, label)
	return model

# color line plots for a single vector
def line_plot_color(data, labels, thresh, names=None):
	# Define colors for each class
	colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
	nClass = np.unique(labels)
	# Plotting each class separately with a different color
	plt.figure(figsize=(8, 3))
	for ii, ll in enumerate(nClass):
		indices = np.where(labels == ll)[0]  # Get indices for each class
		if names is None:
			plt.plot(indices, data[indices], color=colors[ii], label=rf'$F_{ii}$')
		else:
			plt.plot(indices, data[indices], color=colors[ii], label=f'{names[ii]}')
		plt.axhline(thresh, 0, len(data), color='r')
		plt.legend(fontsize=10), plt.title('Recon Error')
		plt.xlabel('Input Samples', fontsize=14), plt.ylabel('MSE', fontsize=14)
	plt.show()
	
# To calculate detection accuracy for each dataset and malware case
def acc_per_class(data, labels, thresh):
	nClass = np.unique(labels)
	acc_results = np.zeros((len(nClass),))
	for ii, ll in enumerate(nClass):
		indices = np.where(labels == ll)[0]  # Get indices for each class
		acc_results[ii] = np.mean(data[indices] > thresh)
	return acc_results

# show latent distribution
def latent_visualize(data, labels, names): # data is the latent variable output of the Encoder
	# latent distribution figures
	colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
	nClass = np.unique(labels)
	if data.shape[1] > 2:
		t0 = time.time()
		data_reduced = TSNE(n_components=2, random_state=42).fit_transform(data)
		t1 = time.time()
		print("T-SNE took {:.2} s".format(t1 - t0))
	else:
		data_reduced = data
	plt.figure(dpi=200)
	plt.grid()
	for ii, ll in enumerate(nClass):
		indices = np.where(labels == ll)[0]
		plt.scatter(data_reduced[indices, 0], data_reduced[indices, 1], label=names[ii], linewidths=0.2,
					edgecolors='none', c=colors[ii])
			
	plt.legend(loc='best', markerscale=2)
	plt.title('Latent Representation')
	plt.xlabel('z(1)', fontsize=14), plt.ylabel('z(2)', fontsize=14), plt.show()

# Wrapper for Binary Classification
def binary_class_wrapper(full_struct):
	names = list(full_struct.keys())
	type_names = ['safe', 'mal1', 'mal2', 'mal3']
	modelNames = ['RFC', 'SVM', 'KNN', 'DT', 'GB']
	# output dataset x numClassifier x Metrics (ACC, TNR, TPR)
	Results = np.zeros((len(names), len(modelNames), 3))
	for ii, key in enumerate(names):
		DS = full_struct[key]
		D_safe = DS[type_names[0]]
		D_mal = np.concatenate(([DS[type_names[1]], DS[type_names[2]], DS[type_names[3]]]), axis=0)
		
		X_data = np.concatenate((D_safe, D_mal), axis=0)
		y_data = np.concatenate((np.zeros((D_safe.shape[0], )), np.ones((X_data.shape[0] - D_safe.shape[0], ))),
								axis=0).astype(np.int16)
		
		X_data = np.reshape(X_data, (X_data.shape[0], -1))
		
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=98,
															shuffle=True, stratify=y_data)
		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		
		# Model Training and results computation
		Results_dataset = np.zeros((len(modelNames), 3))
		for jj, mname in enumerate(modelNames):
			scores = binary_classification(X_train, y_train, X_test, y_test, model=mname)
			Results_dataset[jj, :] = scores
		Results[ii, :, :] = Results_dataset
	return Results

def binary_classification(X_train, y_train, X_test, y_test, model='rfc'):
	if model.lower() == 'rfc':
		model = random_forest_classifier
	elif model.lower() == 'svm':
		model = svm_classifier
	elif model.lower() == 'knn':
		model = knn_classifier
	elif model.lower() == 'dt':
		model = decision_tree_classifier
	elif model.lower() == 'gb':
		model = gradient_boosting_classifier
	else:
		print('Wrong model selected, exiting')
		return 0

	model_trained = model(X_train, y_train)
	print("Results using " + model.__name__ + " On Test dataset")
	scores = perf_rates(y_test, X_test, model=model_trained)
	
	return scores[:3]

# Singular Values plot - Normalized
def plot_singvals(full_struct, lastcomp=150, fskeys=None):
	if fskeys is None:
		fskeys = full_struct.keys()
		
	plt.figure()
	for key in fskeys:
		inner_struct = full_struct[key]
		temp = inner_struct['safe'].reshape((inner_struct['safe'].shape[0], -1))
		U, S, Vh = np.linalg.svd(temp, full_matrices=False)
		cumvar = np.cumsum(S) / np.sum(S) * 100.0
		plt.plot(cumvar[:lastcomp], linewidth=2.5, label=key)
		plt.xlabel('Singular Value Index', fontsize=12)
		plt.ylabel('Variance Explained Percentage', fontsize=12)
		# plt.title(f'Dataset: {key}')
		plt.legend(fontsize=12), plt.grid('ON')


# ROC-AUC curves
def plot_roc(data, labels, names, title='ROC'):
	nClass = np.unique(labels)
	plt.figure()
	for ll in nClass:
		indices = np.where(labels == ll)[0]  # Get indices for each class
		y_true = np.concatenate((np.array([[0]]), np.ones((len(indices), 1))), axis=0).reshape(-1)
		y_out = np.concatenate((np.array([0]), data[indices]))
		fpr, tpr, threshs = roc_curve(y_true, y_out, pos_label=1)
		# Calculate the AUC (Area Under the Curve)
		roc_auc = auc(fpr, tpr)
		# Plot the ROC curve
		# plt.figure(figsize=(8, 6))
		plt.plot(fpr, tpr, lw=2, label=f'{names[ll]}, AUC = {roc_auc:.2f}')
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate (FPR)')
	plt.ylabel('True Positive Rate (TPR)')
	plt.title(f'ROC Curve for {title}')
	plt.legend(loc='lower right')
	plt.show()


# Apply softmax function to logits
def softmax(logits):
	if np.max(logits) > 2:
		logits = logits / np.max(logits)
	exp_logits = np.nan_to_num(np.exp(logits), nan=1.0)
	probabilities = exp_logits / np.sum(exp_logits)
	return probabilities