import torch
import pdb
import numpy as np
from model import *
from utils import *


# Training Parameters
k = 10				# Groups for k-fold validation
width = 1024		# Width of model
epochs = 300		# Training epochs for each fold


# Loading preprocessed data and splitting
Data = np.load("Data.npz")
Data, spec, over, IDs = torch.Tensor(Data['Data']), torch.Tensor(Data['spec']), torch.Tensor(Data['over']), Data['IDs']
groups1 = torch.chunk(torch.randperm(len(IDs)), k)
groups2 = torch.chunk(torch.randperm(len(IDs)), k)


# K-fold validation
for i in range(k):

	# Build k-th specific model, train/test set and train
	spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k)

	spec_model = Model(width)
	spec_optim = torch.optim.Adam(spec_model.parameters(), lr = 3e-4)

	print(f'Training Disease Specific Survival Model #{i} ')
	for epoch in range(epochs):

		# Gather output for train and test set
		spec_prob, spec_pred = spec_model(spec_xtrain)
		
		# Compute loss
		spec_loss = Loss(spec_pred, spec_strain, spec_prob, spec_ptrain, spec_stest, spec_ptest, spec_model, spec_xtest, epoch)
		
		# Compute derivative, step, and reset
		spec_loss.backward()
		spec_optim.step()
		spec_optim.zero_grad()
		
		

	# Build k-th overall model, train/test set and train

	over_xtrain, over_strain, over_ptrain, over_xtest, over_stest, over_ptest = train_and_test_set(Data, over, groups2, i, k)

	over_model = Model(width)
	over_optim = torch.optim.Adam(over_model.parameters(), lr = 3e-4)

	print(f'Training Overall Survival Model #{i} ')
	for epoch in range(epochs):

		# Gather output for train and test set
		over_prob, over_pred = over_model(over_xtrain)

		# Compute loss
		over_loss = Loss(over_pred, over_strain, over_prob, over_ptrain, over_stest, over_ptest, over_model, over_xtest, epoch)

		# Compute derivative, step, and reset
		over_loss.backward()
		over_optim.step()
		over_optim.zero_grad()


	AUCplot(spec_model, over_model, spec_xtest, over_xtest, spec_ptest, over_ptest, k)






