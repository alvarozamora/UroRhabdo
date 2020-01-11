import torch
import pdb
import numpy as np
from model import *

# Training Parameters
k = 10				# Groups for k-fold validation
width = 512			# Width of model
epochs = 5000		# Training epochs for each fold


# Loading preprocessed data and splitting
Data = np.load("Data.npz")
Data, Labels, IDs = torch.Tensor(Data['Data']), torch.Tensor(Data['Labels']), Data['IDs']
groups = torch.chunk(torch.randperm(len(IDs)), k)


# K-fold validation
for i in range(k):

	# Build k-th train and test set
	xtrain = torch.Tensor([])
	strain = torch.Tensor([])
	ptrain = torch.Tensor([])
	for j in range(k):
		if i != j:
			xtrain = torch.cat((xtrain, Data[groups[j]]))
			strain = torch.cat((strain, Labels[groups[j],0]))
			ptrain = torch.cat((ptrain, Labels[groups[j],1]))
		else: 
			xtest = Data[groups[j]]
			stest = Labels[groups[j],0]
			ptest = Labels[groups[j],1]


	model = Model(width)
	optim = torch.optim.Adam(model.parameters(), lr = 3e-4)

	for epoch in range(epochs):

		prob, pred = model(xtrain)
		tprob, tpred = model(xtest)

		mse  = F.mse_loss(pred, strain)
		bce  = F.binary_cross_entropy(prob, ptrain)
		tmse = F.mse_loss(tpred, stest)
		tbce = F.binary_cross_entropy(tprob, ptest)

		loss  = mse + bce
		tloss = tmse + tbce
		
		loss.backward()
		optim.step()
		optim.zero_grad()

		acc = (pred > 0.5).float().mean()
		tacc = (tpred > 0.5).float().mean()
		if (epoch+1)%100 == 0:
			print(f'epoch = {epoch+1}; mse = {mse:.3f}; bce = {bce:.3f}; tmse = {tmse:.3f}; tbce = {tbce:.3f}; acc = {acc:.3f}; test acc = {tacc:.3f}')





