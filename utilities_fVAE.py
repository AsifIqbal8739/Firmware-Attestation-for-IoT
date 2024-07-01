import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle

import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.io import loadmat
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the VAE Model
class Model_VAE(nn.Module):
	def __init__(self, config):
		super(Model_VAE, self).__init__()
		self.config = config
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.input_dim = config.input_dim
		self.label_dim = config.label_dim  # label encoding dimension
		self.enc_hid_dims = config.enc_hid_dims
		self.latent_dims = config.latent_dims
		self.dec_hid_dims = config.dec_hid_dims
		
		# Encoder Layers for VAE
		encdims = [self.input_dim + self.label_dim] + self.enc_hid_dims     # list of nodes
		self.enc_modules = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in
										  zip(encdims[:-1], encdims[1:])])
		self.enc_mean = nn.Linear(encdims[-1], self.latent_dims)
		self.enc_logvar = nn.Linear(encdims[-1], self.latent_dims)
		self.enc_modules.append(self.enc_mean).append(self.enc_logvar)
		
		# Decoder Layers for VAE
		decdims = [self.latent_dims + self.label_dim] + self.dec_hid_dims + [self.input_dim]
		self.dec_modules = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in
										  zip(decdims[:-1], decdims[1:])])

		self.activation = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		# Initialization
		self.xavier_init()
	
	def encode(self, x):
		for ll, layer in enumerate(self.enc_modules):
			if ll < (len(self.enc_modules) - 2):
				x = self.activation(layer(x))
			elif ll == (len(self.enc_modules) - 2):
				z_mean = self.enc_mean(x)
			else:
				z_logvar = self.enc_logvar(x)
		return z_mean, z_logvar
	
	def reparam(self, z_mean, z_logvar):
		std = torch.exp(0.5 * z_logvar)
		eps = torch.randn_like(std)
		return z_mean + std * eps
	
	def decode(self, z):
		for ll, layer in enumerate(self.dec_modules):
			if ll < (len(self.dec_modules) - 1):
				z = self.activation(layer(z))
			else:
				z = layer(z)
		return z
	
	def forward(self, x, labels=None, clip=False): # During training clip is False
		if self.label_dim > 0:
			x = torch.concatenate((x, labels), dim=1).to(torch.float32)
		z_mean, z_logvar = self.encode(x)
		if clip:
			z_mean = torch.clamp(z_mean, min=-clip, max=clip)
			z_logvar = torch.clamp(z_logvar, min=-np.log(clip), max=np.log(clip))
		z = self.reparam(z_mean, z_logvar)
		if self.label_dim > 0:
			z = torch.concatenate((z, labels), dim=1).to(torch.float32)
		x_recon = self.decode(z)
		return x_recon, z_mean, z_logvar
	
	def xavier_init(self):
		modulelist = nn.ModuleList([self.enc_modules, self.dec_modules])
		for modules in modulelist:
			for layer in modules:
				if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
					nn.init.xavier_uniform_(layer.weight)
					if layer.bias is not None:
						nn.init.zeros_(layer.bias)
		
		
# Training class
class Exp_VAE(object):
	def __init__(self, config):
		self.config = config
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model = self._build_model().to(self.device)
		self.scaler = MinMaxScaler() # MinMaxScaler(), DummyScaler
		self.onehotencoder = OneHotEncoder()
		self.batchSize = config.batchSize
		self.lr = config.lr  # A list of lr for a number of epochs
		self.epochs = config.epochs
		self.lrSwitch = config.lrSwitch
		self.conditional = 1 if config.label_dim else 0  # Conditional VAE?
		
		# parameters controlling training within VAE
		self.latent_dims = config.latent_dims
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr[0])
		self.rec_loss = nn.MSELoss()
		
	def _build_model(self):
		model = Model_VAE(self.config)
		print('CVAE Model Created.')
		return model
	
	def loss_function(self, x, x_recon, z_mean, z_logvar):
		recon_loss = self.rec_loss(x, x_recon)
		# recon_loss = nn.functional.l1_loss(x, x_recon, reduction='mean')
		kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
		# print(f'Recon Loss: {recon_loss}, KL Loss: {kl_loss}')
		return recon_loss, kl_loss
	
	def _get_data(self):
		self.X = self.config.X_data
		# y = np.zeros((self.X.shape[0],), dtype=np.int8)
		self.Y = self.config.Y_label.reshape(-1, 1)
		
		# One Hot Encoding
		y_train = self.onehotencoder.fit_transform(self.Y).toarray()
		
		X_train = self.scaler.fit_transform(self.X)
		# self.scaler.scale_[self.scaler.scale_ > 255] = 255      # SPECIAL Scaling to control value blowup
		
		# Convert data to PyTorch tensors
		self.X_train = torch.from_numpy(X_train).float().to(self.device)
		self.y_train = torch.from_numpy(y_train).float().to(self.device)
		
		# Create DataLoader objects for batching
		train_dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True)
		
		return train_loader
	
	def train(self, verbose=True):
		train_loader = self._get_data()
		recon_ll, kl_ll = [], []
		for epoch in range(self.epochs):
			rl, kll = 0.0, 0.0  # running losses
			for ii, (inputs, labels) in enumerate(train_loader, start=0):
				self.optimizer.zero_grad()
				recon, z_mean, z_logvar = self.model(inputs, labels=labels)
				recon_loss, kl_loss = self.loss_function(inputs, recon, z_mean, z_logvar)
				loss = recon_loss + kl_loss
				loss.backward()
				self.optimizer.step()
				
				# running losses
				rl += recon_loss.item()
				kll += kl_loss.item()
			
				# print progress
				if ii % 5 == 0 and verbose:  # Print every 10 batches
					print(f"Epoch [{epoch}/{self.epochs}] | Batch [{ii}/{len(train_loader)}] | "
						  f"Recon Loss: {recon_loss.item():.4f} | KL Loss: {kl_loss.item():.4f} |")
			recon_ll.append(rl)
			kl_ll.append(kll)
			
			# Change the learning rate
			self._update_lr(epoch)
		return recon_ll, kl_ll
		
	def test(self):
		with torch.no_grad():
			recon = self.model(self.X_test)[0]
			loss = nn.functional.mse_loss(self.X_test, recon, reduction='mean')
		print("---------Test loss: MSE: {:.4f}--------------".format(loss.item()))
		
	# learning rate change helper
	def _update_lr(self, epoch):
		if epoch == self.lrSwitch[0]:
			newlr = self.lr[1]
		elif epoch == self.lrSwitch[1]:
			newlr = self.lr[2]
		# self.lambd = 1
		if epoch == self.lrSwitch[0] or epoch == self.lrSwitch[1]:
			for param in self.optimizer.param_groups:
				param['lr'] = newlr
			print(f"\n\t Learning rate updated to {newlr}")
			
			
# Latent feature extraction of input data - Recon using reparam latent variable
# 'clip' the abs values to a specific value
def ext_outputs(exp_main, data=None, labels=None, clip=False):   # None for trained data
	if data is None:    # using the training data
		data = exp_main.X_train        # Non transformed data
		labels = exp_main.y_train
	if not torch.is_tensor(data):
		data = torch.from_numpy(exp_main.scaler.transform(data)).float().to(exp_main.device)
		if labels is not None:
			labels = torch.from_numpy(exp_main.onehotencoder.transform(labels.reshape(-1, 1)).toarray()).to(exp_main.device)
	if clip:
		data = torch.clamp(data, min=-np.abs(clip), max=np.abs(clip))
	with torch.no_grad():
		# z_mean, z_logvar = exp_main.model.encode(data)
		# z_latent = exp_main.model.reparam(z_mean, z_logvar)
		# recon = exp_main.model.decode(z_latent)
		recon, z_mean, z_logvar = exp_main.model(data, labels=labels)
		recon_error = torch.mean((data - recon)**2, dim=1)
	z_mean = z_mean.detach().cpu().numpy()
	z_logvar = z_logvar.detach().cpu().numpy()
	recon = exp_main.scaler.inverse_transform(recon.detach().cpu().numpy())
	recon_error = recon_error.detach().cpu().numpy()

	return recon, recon_error, z_mean, z_logvar

class DummyScaler:
	def __init__(self):
		pass

	def fit(self, X):
		# Dummy method that does nothing
		return self

	def transform(self, X):
		# Dummy method that returns the input unchanged
		return X

	def fit_transform(self, X):
		# Dummy method that combines fit and transform (returns the input unchanged)
		return X
	
	def inverse_transform(self, X):
		# Dummy method that returns the input unchanged
		return X