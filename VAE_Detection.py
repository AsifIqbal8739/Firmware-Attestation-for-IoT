# Multi-Trial Mean results for CVAE
import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import utilities_firmware as utf
import utilities_fVAE as utvae
import torch
from torch.utils.data import DataLoader
import time
import argparse
from importlib import reload
import warnings
warnings.filterwarnings("ignore")
reload(utvae)

file_path = '.\\Data\\firmwares.pkl'

# Load the dictionary from the file
with open(file_path, 'rb') as file:
	full_struct = pickle.load(file)
	
# Normalizing all data to 0 -> 0 and 255.0 -> 1 ##########################################
utf.struct_normalize(full_struct)   #

num_datasets = len(full_struct.keys())
# dataset_names = list(full_struct.keys())
type_names = ['safe', 'mal1', 'mal2', 'mal3']
dataset_names = ['AES128', 'Interrupt', 'LED', 'Random', 'Shake', 'Temperature', 'Vibration', 'XTS']

# Type of pre-processing selection below
ppType = 3  # type 3 represents the proposed prep pipeline from the paper
if ppType == 1:
	full_struct_mod = full_struct
elif ppType == 2:
	full_struct_mod = utf.remove_comp(full_struct, method='pow', kcomp=1)
elif ppType == 3:
	full_struct_mod = utf.keep_comp(full_struct, kcomp=np.arange(1, 200), scale=False)
elif ppType == 4:
	full_struct_mod = utf.hog_prep(full_struct)

# Separate all datasets for training and testing
Safe_DS, Mal1_DS, Mal2_DS, Mal3_DS = [], [], [], []
Safe_DS_L, Mal1_DS_L, Mal2_DS_L, Mal3_DS_L = [], [], [], []
for ii, dsname in enumerate(dataset_names):
	Safe_DS = Safe_DS + [full_struct_mod[dsname]['safe'].reshape((full_struct_mod[dsname]['safe'].shape[0], -1))]
	Safe_DS_L = Safe_DS_L + [ii * np.ones((full_struct_mod[dsname]['safe'].shape[0], ), dtype=int)]
	Mal1_DS = Mal1_DS + [full_struct_mod[dsname]['mal1'].reshape((full_struct_mod[dsname]['mal1'].shape[0], -1))]
	Mal1_DS_L = Mal1_DS_L + [ii * np.ones((full_struct_mod[dsname]['mal1'].shape[0], ), dtype=int)]
	Mal2_DS = Mal2_DS + [full_struct_mod[dsname]['mal2'].reshape((full_struct_mod[dsname]['mal2'].shape[0], -1))]
	Mal2_DS_L = Mal2_DS_L + [ii * np.ones((full_struct_mod[dsname]['mal2'].shape[0], ), dtype=int)]
	Mal3_DS = Mal3_DS + [full_struct_mod[dsname]['mal3'].reshape((full_struct_mod[dsname]['mal3'].shape[0], -1))]
	Mal3_DS_L = Mal3_DS_L + [ii * np.ones((full_struct_mod[dsname]['mal3'].shape[0], ), dtype=int)]

Train_DS, Train_DS_Label = np.concatenate(Safe_DS, axis=0), np.concatenate(Safe_DS_L, axis=0)
Mal1_DS, Mal1_DS_Label = np.concatenate(Mal1_DS, axis=0), np.concatenate(Mal1_DS_L, axis=0)
Mal2_DS, Mal2_DS_Label = np.concatenate(Mal2_DS, axis=0), np.concatenate(Mal2_DS_L, axis=0)
Mal3_DS, Mal3_DS_Label = np.concatenate(Mal3_DS, axis=0), np.concatenate(Mal3_DS_L, axis=0)

# Configuration setting for the VAE model training
dict = {}
dict['X_data'] = Train_DS
dict['Y_label'] = Train_DS_Label
dict['label_dim'] = len(dataset_names)
# Main input model parameters
dict['input_dim'] = Train_DS.shape[1]
dict['enc_hid_dims'] = [100, 50]
dict['latent_dims'] = 5
dict['dec_hid_dims'] = [50, 100]
dict['test_size'] = 0.05

# Inner model parameters
dict['batchSize'] = 512
dict['epochs'] = 200
dict['lr'] = [1e-4, 5e-5, 1e-5]
dict['lrSwitch'] = [100, 150]

# Number of Trials for mean scores
nTrials = 10
Acc_Recon, Acc_Latent = 0, 0
for nT in range(nTrials):
	# Training the Model
	config = argparse.Namespace(**dict)
	exp_main = utvae.Exp_VAE(config)
	
	re_ll, kl_ll = exp_main.train()
	# plt.figure(), plt.plot(re_ll[5:], label='Recon Loss'), plt.plot(kl_ll[5:], label='KL Loss')
	# plt.legend(), plt.xlabel('Epochs')
	
	# internal outputs for training performance check
	recon, rec_error, z_mean, z_logvar = utvae.ext_outputs(exp_main=exp_main)
	
	# Threshold Compute
	FPR = 1e-3
	Thresh = np.percentile(rec_error, 100*(1-FPR))
	Thresh_latent = np.percentile(np.linalg.norm(z_mean, axis=1), 100*(1-FPR))
	
	# Computing the decision variables with Clipped latent variables during inference
	clipVal = 2.0
	recon_m1, rec_error_m1, z_mean_m1, z_logvar_m1 = utvae.ext_outputs(exp_main=exp_main, data=Mal1_DS, labels=Mal1_DS_Label, clip=clipVal)
	recon_m2, rec_error_m2, z_mean_m2, z_logvar_m2 = utvae.ext_outputs(exp_main=exp_main, data=Mal2_DS, labels=Mal2_DS_Label, clip=clipVal)
	recon_m3, rec_error_m3, z_mean_m3, z_logvar_m3 = utvae.ext_outputs(exp_main=exp_main, data=Mal3_DS, labels=Mal3_DS_Label, clip=clipVal)
	
	#-----------------Accuracy for each class withing Mal1 2 and 3 WRT Recon and Latent Distribution----------------
	Acc_Recon_Mal1 = utf.acc_per_class(rec_error_m1, Mal1_DS_Label, Thresh)     # biased accuracy
	Acc_Recon_Mal2 = utf.acc_per_class(rec_error_m2, Mal2_DS_Label, Thresh)
	Acc_Recon_Mal3 = utf.acc_per_class(rec_error_m3, Mal3_DS_Label, Thresh)
	Acc_Recon += np.vstack([Acc_Recon_Mal1, Acc_Recon_Mal2, Acc_Recon_Mal3])
	np.mean(Acc_Recon, axis=1)
	
	Acc_Latent_Mal1 = utf.acc_per_class(np.linalg.norm(z_mean_m1, axis=1), Mal1_DS_Label, Thresh_latent)
	Acc_Latent_Mal2 = utf.acc_per_class(np.linalg.norm(z_mean_m2, axis=1), Mal2_DS_Label, Thresh_latent)
	Acc_Latent_Mal3 = utf.acc_per_class(np.linalg.norm(z_mean_m3, axis=1), Mal3_DS_Label, Thresh_latent)
	Acc_Latent += np.vstack([Acc_Latent_Mal1, Acc_Latent_Mal2, Acc_Latent_Mal3])
	np.mean(Acc_Latent, axis=1)
	
Acc_comb = np.vstack([Acc_Recon/nTrials, Acc_Latent/nTrials])*100
np.savetxt(f'AccRes_{ppType}.csv', Acc_comb, delimiter=',', fmt='%.5f')
