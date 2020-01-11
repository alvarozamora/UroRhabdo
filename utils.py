import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def train_and_test_set(data, labels, groups, i, k):

	# Specific
	spec_xtrain = torch.Tensor([])
	spec_strain = torch.Tensor([])
	spec_ptrain = torch.Tensor([])
	for j in range(k):
		if i != j:
			spec_xtrain = torch.cat((spec_xtrain, data[groups[j]]))
			spec_strain = torch.cat((spec_strain, labels[groups[j],0]))
			spec_ptrain = torch.cat((spec_ptrain, labels[groups[j],1]))
		else: 
			spec_xtest = data[groups[j]]
			spec_stest = labels[groups[j],0]
			spec_ptest = labels[groups[j],1]

	return spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest

def Loss(pred, strain, prob, ptrain, stest, ptest, model, xtest, epoch):

	mse_scale = 1e5

	mse  = F.mse_loss(pred, strain)/mse_scale
	bce  = F.binary_cross_entropy(prob, ptrain)

	loss  = mse + bce

	#Evaluate
	if (epoch+1)%100 == 0:

		model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
		tprob, tpred = model(xtest)
		model.train() #This turns BatchNorm and Dropout back on

		tmse = F.mse_loss(tpred, stest)/mse_scale
		tbce = F.binary_cross_entropy(tprob, ptest)
		
		tloss = tmse + tbce

		acc = (pred > 0.5).float().mean()
		tacc = (tpred > 0.5).float().mean()

		auc, tauc = AUC(ptrain.data.numpy(), ptest.data.numpy(), prob.data.numpy(), tprob.data.numpy())


		print(f'epoch = {epoch+1}; mse = {mse:.3f}; bce = {bce:.3f}; tmse = {tmse:.3f}; tbce = {tbce:.3f}; acc = {acc:.3f}; test acc = {tacc:.3f}; AUC = {auc:.2f}, test AUC = {tauc:2f}')

	return loss


def AUC(ytrain, ytest, prob, tprob):
	AUCtrain = roc_auc_score(ytrain, prob)
	AUCtest  = roc_auc_score(ytest, tprob)

	return AUCtrain, AUCtest


def AUCplot(spec_model, over_model, spec_xtest, over_xtest, yspec, yover, k):

	spec_model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
	over_model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
	probspec, _ = spec_model(spec_xtest)
	probover, _ = over_model(over_xtest)
	spec_model.train() #This turns BatchNorm and Dropout back on
	over_model.train() #This turns BatchNorm and Dropout back on

	fpr_spec, tpr_spec, thresholds_spec = roc_curve(yspec.data.numpy(), probspec.data.numpy())
	fpr_over, tpr_over, thresholds_over = roc_curve(yover.data.numpy(), probover.data.numpy())


	AUCspec = roc_auc_score(yspec.data.numpy(), probspec.data.numpy())
	AUCover = roc_auc_score(yover.data.numpy(), probover.data.numpy())

	plt.figure()
	plt.plot(fpr_spec, tpr_spec, label = f'Specific = {AUCspec:.2f}')
	plt.plot(fpr_over, tpr_over, label =  f'Overall = {AUCover:.2f}')
	plt.plot([0,1],[0,1],'k-')
	plt.grid(alpha=0.2)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(title='AUC')
	plt.savefig(f'ROCs/ROC_{k}.png')

	print(f'\n\nGenerated Specific vs Overall ROC plot for test set #{k} using lastest models in ./ROCs/\n\n')






