import torch
import pdb
import copy
import numpy as np
from LSUV import LSUVinit
from model import *
from utils import *
from pytorch2keras import pytorch_to_keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model as KModel
import tensorflow.keras.backend as KB


# Training Parameters
k = 5				# Groups for k-fold validation
width = 256		# Width of model
depth = 8			# Depth of model
epochs = 5000		# Training epochs for each fold
L2 = 1e-4			# L2 Regularization
L1 = 1e-4			# L1 Regularization
LR = 3e-4			# Learning Rate
minauc = 0.85


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading preprocessed data and splitting
Data = np.load("Data.npz")
Data, spec, over, IDs = torch.Tensor(Data['Data']), torch.Tensor(Data['spec']), torch.Tensor(Data['over']), Data['IDs']
groups1 = torch.chunk(torch.randperm(len(IDs)), k)
#groups2 = groups1 #torch.chunk(torch.randperm(len(IDs)), k)
groups2 = torch.chunk(torch.randperm(len(IDs)), k)

# Check for One-Class Sets
spec_check = True
over_check = True
while spec_check:
	for i in range(k):
		spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k, device)
	
		print(f'Specific fraction of positives: fold {i}; train {spec_ptrain.sum()/len(spec_ptrain):.3f}; test {spec_ptest.sum()/len(spec_ptest):.3f}')
		if spec_ptrain.sum() == len(spec_ptrain) or spec_ptest.sum() == len(spec_ptest):
			print(f'One Class Set (Specific, k = {i})')
			groups1 = torch.chunk(torch.randperm(len(IDs)), k)
			spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k, device)
			break

	spec_check = False

while over_check:
	for i in range(k):
		over_xtrain, over_strain, over_ptrain, over_xtest, over_stest, over_ptest = train_and_test_set(Data, over, groups1, i, k, device)
	
		print(f'Overall fraction of positives: fold {i}; train {over_ptrain.sum()/len(over_ptrain):.3f}; test {over_ptest.sum()/len(over_ptest):.3f}')
		if over_ptrain.sum() == len(over_ptrain) or over_ptest.sum() == len(over_ptest):
			print(f'One Class Set (Overall, k = {i})')
			groups1 = torch.chunk(torch.randperm(len(IDs)), k)
			over_xtrain, over_strain, over_ptrain, over_xtest, over_stest, over_ptest = train_and_test_set(Data, over, groups1, i, k, device)
			break

	over_check = False

# Lists for collecting ROCs and Calibration Curves
spec_rocs = []
spec_cals = []
over_rocs = []
over_cals = []


spec_models = []
over_models = []
# K-fold validation
for i in range(k):

	# Build models
	#spec_model = Model(width)
	

	# Build Specific train/test set
	spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k, device)
	while spec_ptrain.sum() == len(spec_ptrain) or spec_ptest.sum() == len(spec_ptest):
		print("One Class Set")
		spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest = train_and_test_set(Data, spec, groups1, i, k, device)
	best_spec = 0

	#model = LSUVinit(model, spec_xtrain, needed_std = 1.0, cuda = torch.cuda.is_available())
	
	while best_spec < minauc:
		spec_model = Model(width,depth)#LSUVinit(Model(width, depth), spec_xtrain)
		spec_optim = torch.optim.Adam(spec_model.parameters(), lr = LR, weight_decay = L2)
		


		print(f'Training Disease Specific Survival Model #{i} ')
		for epoch in range(epochs):



			# Gather output for train and test set
			spec_prob, spec_pred = spec_model(spec_xtrain)
		
			# Compute loss
			spec_loss, spec_auc, spec_tauc = Loss(spec_pred, spec_strain, spec_prob, spec_ptrain, spec_stest, spec_ptest, spec_model, spec_xtest, epoch, best_spec, L1)
			if spec_tauc > best_spec and spec_auc >= spec_tauc:
				#print("New Best Specific")
				best_spec = spec_tauc
				best_spec_model = copy.deepcopy(spec_model)

			# Compute derivative, step, and reset
			spec_loss.backward()
			spec_optim.step()
			spec_optim.zero_grad()
		print(f'Best Specific Model has test AUC = {best_spec:.3f}')

	spec_models.append(copy.deepcopy(best_spec_model))
	

	# Build Overall train/test set and train

	over_xtrain, over_strain, over_ptrain, over_xtest, over_stest, over_ptest = train_and_test_set(Data, over, groups2, i, k, device)

	#over_model = Model(width)
	best_over = 0

	while best_over < minauc:
		over_model = Model(width,depth)#LSUVinit(Model(width, depth), over_xtrain)
		over_optim = torch.optim.Adam(over_model.parameters(), lr = LR, weight_decay = L2)
		

		print(f'Training Overall Survival Model #{i}')
		for epoch in range(epochs):


			# Gather output for train and test set
			over_prob, over_pred = over_model(over_xtrain)

			# Compute loss
			over_loss, over_auc, over_tauc = Loss(over_pred, over_strain, over_prob, over_ptrain, over_stest, over_ptest, over_model, over_xtest, epoch, best_over, L1)
			if over_tauc > best_over and over_auc >= over_tauc:
				#print("New Best Overall")
				best_over = over_tauc
				best_over_model = copy.deepcopy(over_model)

			# Compute derivative, step, and reset
			over_loss.backward()
			over_optim.step()
			over_optim.zero_grad()
		print(f'Best Overall Model has test AUC = {best_over:.3f}')

	over_models.append(copy.deepcopy(best_over_model))

	spec_rocs, over_rocs, spec_cals, over_cals = AUCplot(best_spec_model, best_over_model, spec_xtest, over_xtest, spec_ptest, over_ptest, spec_rocs, over_rocs, spec_cals, over_cals, i, k)

#final_spec = CombinedModel(spec_models)
final_specs = nn.ModuleList([nn.Sequential(model.trunk, model.prob, nn.Sigmoid()) for model in spec_models])
final_overs = nn.ModuleList([nn.Sequential(model.trunk, model.prob, nn.Sigmoid()) for model in over_models])

print("Keras-ifying Models")


k_spec = []
k_over = []
for i in range(len(final_specs)):
	final_specs[i].eval()
	spec_model_k = pytorch_to_keras(final_specs[i], spec_xtrain, verbose=True)
	k_spec.append(spec_model_k)
	spec_model_k.save(f'Final_DSS_{i:03d}.h5')
	torch.save(final_specs[i], f'Final_DSS_{i:03d}.pth')
for i in range(len(final_overs)):
	final_overs[i].eval()
	over_model_k = pytorch_to_keras(final_overs[i], over_xtrain, verbose=True)
	k_over.append(over_model_k)
	over_model_k.save(f'Final_OS_{i:03d}.h5')
	torch.save(final_specs[i], f'Final_OS_{i:03d}.pth')

#pdb.set_trace()
spec_input = Input(shape=(9,))
spec_output = KB.mean(KB.concatenate([spec(spec_input) for spec in k_spec], axis=0), axis=0)

spec_combined_model_k = KModel(inputs=spec_input, outputs=spec_output)
spec_combined_model_k.save(f'Final_DSS.h5')

over_input = Input(shape=(9,))
over_output = KB.mean(KB.concatenate([over(over_input) for over in k_over], axis=0), axis=0)

over_combined_model_k = KModel(inputs=over_input, outputs=over_output)
spec_combined_model_k.save(f'Final_OS.h5')

#spec_model_k = pytorch_to_keras(final_spec, spec_xtrain, verbose=True)#, (9,), verbose=True, names='short')  
#final_over = CombinedModel(over_models)
#orch.save(final_over, 'Combined_Overall.pth')






