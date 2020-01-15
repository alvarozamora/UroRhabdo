import torch
import pdb
import copy
import numpy as np
from LSUV import LSUVinit
from model import *
from utils import *


# Training Parameters
k = 5				# Groups for k-fold validation
width = 256			# Width of model
epochs = 3000		# Training epochs for each fold
L2 = 1e-3			# L2 Regularization
L1 = 1e-3			# L1 Regularization
LR = 1e-3			# Learning Rate

# Loading preprocessed data and splitting
Data = np.load("Data.npz")
Data, spec, over, IDs = torch.Tensor(Data['Data']), torch.Tensor(Data['spec']), torch.Tensor(Data['over']), Data['IDs']
groups1 = torch.chunk(torch.randperm(len(IDs)), k)
#groups2 = groups1 #torch.chunk(torch.randperm(len(IDs)), k)
groups2 = torch.chunk(torch.randperm(len(IDs)), k)

# Lists for collecting ROCs and Calibration Curves
spec_rocs = []
spec_cals = []
over_rocs = []
over_cals = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(width).to(device)

# K-fold validation
for i in range(k):

	# Build models
	#spec_model = Model(width)
	

	# Build Specific train/test set
	spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k, device)
	best_spec = 0

	print(f'Training Disease Specific Survival Model #{i} ')
	for epoch in range(epochs):

		if epoch == 0 and i == 0:
			model = LSUVinit(model, spec_xtrain, needed_std = 1.0, cuda = True)
			spec_model = copy.deepcopy(model)
			over_model = copy.deepcopy(model)
			spec_optim = torch.optim.Adam(spec_model.parameters(), lr = LR, weight_decay = L2)
			over_optim = torch.optim.Adam(over_model.parameters(), lr = LR, weight_decay = L2)
		elif epoch == 0:
			spec_model = copy.deepcopy(model)
			over_model = copy.deepcopy(model)
			spec_optim = torch.optim.Adam(spec_model.parameters(), lr = LR, weight_decay = L2)
			over_optim = torch.optim.Adam(over_model.parameters(), lr = LR, weight_decay = L2)



		# Gather output for train and test set
		spec_prob, spec_pred = spec_model(spec_xtrain)
		
		# Compute loss
		spec_loss, spec_tauc = Loss(spec_pred, spec_strain, spec_prob, spec_ptrain, spec_stest, spec_ptest, spec_model, spec_xtest, epoch, best_spec, L1)
		if spec_tauc > best_spec:
			#print("New Best Specific")
			best_spec = spec_tauc
			best_spec_model = copy.deepcopy(spec_model)

		# Compute derivative, step, and reset
		spec_loss.backward()
		spec_optim.step()
		spec_optim.zero_grad()
	print(f'Best Specific Model has test AUC = {best_spec:.3f}')
	

	# Build Overall train/test set and train

	over_xtrain, over_strain, over_ptrain, over_xtest, over_stest, over_ptest = train_and_test_set(Data, over, groups2, i, k, device)

	#over_model = Model(width)
	best_over = 0

	print(f'Training Overall Survival Model #{i}')
	for epoch in range(epochs):


		# Gather output for train and test set
		over_prob, over_pred = over_model(over_xtrain)

		# Compute loss
		over_loss, over_tauc = Loss(over_pred, over_strain, over_prob, over_ptrain, over_stest, over_ptest, over_model, over_xtest, epoch, best_over, L1)
		if over_tauc > best_over:
			#print("New Best Overall")
			best_over = over_tauc
			best_over_model = copy.deepcopy(over_model)

		# Compute derivative, step, and reset
		over_loss.backward()
		over_optim.step()
		over_optim.zero_grad()
	print(f'Best Overall Model has test AUC = {best_over:.3f}')


	spec_rocs, over_rocs, spec_cals, over_cals = AUCplot(best_spec_model, best_over_model, spec_xtest, over_xtest, spec_ptest, over_ptest, spec_rocs, over_rocs, spec_cals, over_cals, i, k)





