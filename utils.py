import pdb
import torch
import numpy as np
from scipy import interp
import torch.nn.functional as F
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score



def train_and_test_set(data, labels, groups, i, k, device):

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

	spec_xtrain = spec_xtrain.to(device)
	spec_strain = spec_strain.to(device)
	spec_ptrain = spec_ptrain.to(device)
	spec_xtest = spec_xtest.to(device)
	spec_stest = spec_stest.to(device)
	spec_ptest = spec_ptest.to(device)
	return spec_xtrain, spec_strain, spec_ptrain, spec_xtest, spec_stest, spec_ptest

def Loss(pred, strain, prob, ptrain, stest, ptest, model, xtest, epoch, best, L1):

	mse_scale = 1.0

	mse  = F.mse_loss(pred, strain)/mse_scale
	bce  = F.binary_cross_entropy(prob, ptrain)

	loss  = mse + bce + L1*model.L1_Loss()

	#Evaluate
	if (epoch+1)%1 == 0:

		model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
		tprob, tpred = model(xtest)
		model.train() #This turns BatchNorm and Dropout back on

		tmse = F.mse_loss(tpred, stest)/mse_scale
		tbce = F.binary_cross_entropy(tprob, ptest)
		
		tloss = tmse + tbce

		acc = (pred > 0.5).float().mean()
		tacc = (tpred > 0.5).float().mean()

		auc, tauc = AUC(ptrain.data.cpu().numpy(), ptest.data.cpu().numpy(), prob.data.cpu().numpy(), tprob.data.cpu().numpy())

	if (epoch+1)%100 == 0:
		print(f'epoch = {epoch+1}; mse = {mse:.3f}; bce = {bce:.3f}; tmse = {tmse:.3f}; tbce = {tbce:.3f}; acc = {acc:.3f}; test acc = {tacc:.3f}; AUC = {auc:.4f}; test AUC = {tauc:.4f}, best test AUC = {best:.3f}')

	return loss, tauc


def AUC(ytrain, ytest, prob, tprob):
	AUCtrain = roc_auc_score(ytrain, prob)
	AUCtest  = roc_auc_score(ytest, tprob)

	return AUCtrain, AUCtest


def AUCplot(spec_model, over_model, spec_xtest, over_xtest, yspec, yover, spec_rocs, over_rocs, spec_cals, over_cals, k, K):

	# Gather Probabilities from Models
	spec_model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
	over_model.eval() # This turns off BatchNorm and Dropout, as will be done at deployment
	probspec, _ = spec_model(spec_xtest)
	probover, _ = over_model(over_xtest)
	spec_model.train() #This turns BatchNorm and Dropout back on
	over_model.train() #This turns BatchNorm and Dropout back on

	# Generate ROC Curves
	fpr_spec, tpr_spec, thresholds_spec = roc_curve(yspec.data.cpu().numpy(), probspec.data.cpu().numpy())
	fpr_over, tpr_over, thresholds_over = roc_curve(yover.data.cpu().numpy(), probover.data.cpu().numpy())
	spec_rocs.append([fpr_spec, tpr_spec])
	over_rocs.append([fpr_over, tpr_over])

	# Generate Calibration Curves
	N = 4
	strategy = 'uniform'
	spec_cal  = calibration_curve(yspec.data.cpu().numpy(), probspec.data.cpu().numpy(), n_bins=N, strategy=strategy)
	over_cal  = calibration_curve(yspec.data.cpu().numpy(), probspec.data.cpu().numpy(), n_bins=N, strategy=strategy)
	assert ((len(spec_cal[1])==N) and (len(over_cal[1])==N)), f'len is {len(spec_cal[1])}, {len(over_cal[1])} not {N}'
	spec_cals.append(spec_cal)
	over_cals.append(over_cal)

	# Compute AUC Scores
	AUCspec = roc_auc_score(yspec.data.cpu().numpy(), probspec.data.cpu().numpy())
	AUCover = roc_auc_score(yover.data.cpu().numpy(), probover.data.cpu().numpy())


	# Plot k-th Model Performance (ROC)
	plt.figure(0)
	plt.clf()
	plt.plot(fpr_spec, tpr_spec, label = f'Specific = {AUCspec:.2f}')#, alpha = 0.2)
	plt.plot(fpr_over, tpr_over, label =  f'Overall = {AUCover:.2f}')#, alpha = 0.2)
	if k == 0:
		plt.plot([0,1],[0,1],'k-', alpha = 0.2)
	plt.grid(alpha=0.2)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(title='AUC')
	plt.savefig(f'ROCs/ROC_{k}.png')


	if (k+1) == K:
		# Plot Mean Model Performance (ROC)
		mean_fpr = np.linspace(0,1,10000)

		mean_spec_roc = []

		for ROC in spec_rocs:
   			 mean_spec_tpr = interp(mean_fpr, ROC[0], ROC[1])
   			 mean_spec_tpr[0] = 0.0
   			 mean_spec_roc.append(mean_spec_tpr)

		mean_spec_roc = np.array(mean_spec_roc)

		std_tpr = np.std(mean_spec_roc, axis=0)
		mean_spec_roc = np.mean(mean_spec_roc, axis=0)
		mean_spec_roc[-1] = 1.0
		spec_top = np.minimum(mean_spec_roc + std_tpr, 1)
		spec_bot = np.maximum(mean_spec_roc - std_tpr, 0)
		#pdb.set_trace()
		#pdb.set_trace()
		
		mean_auc = trapz(mean_spec_roc, mean_fpr)
		aucs = [trapz(ROC[1], ROC[0]) for ROC in spec_rocs]
		std_auc = np.std(aucs)

		spec_fig, spec_ax = plt.subplots()
		

		spec_ax.plot(mean_fpr, mean_spec_roc, color ='C0', label=f'Mean ROC, AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
		for ROC in spec_rocs:
			spec_ax.plot(ROC[0], ROC[1], alpha=0.2)
		spec_ax.fill_between(mean_fpr, spec_bot, spec_top, color='C0', alpha=.1, label=f'$\pm$ 1 std')
		spec_ax.grid(alpha=0.2)
		spec_ax.legend()
		spec_ax.set_xlabel('False Positive Rate')
		spec_ax.set_ylabel('True Positive Rate')
		spec_ax.title.set_text("Disease Specific Survival Model")
		plt.savefig('DSS_Mean_ROC.png', dpi = 230)



		mean_over_roc = []

		for ROC in over_rocs:
   			 mean_over_tpr = interp(mean_fpr, ROC[0], ROC[1])
   			 mean_over_tpr[0] = 0.0
   			 mean_over_roc.append(mean_over_tpr)

		mean_over_roc = np.array(mean_over_roc)
		std_tpr = np.std(mean_over_roc, axis=0)
		mean_over_roc = np.mean(mean_over_roc, axis=0)
		mean_over_roc[-1] = 1.0
		over_top = np.minimum(mean_over_roc + std_tpr, 1) 
		over_bot = np.maximum(mean_over_roc - std_tpr, 0) 

		
		mean_auc = trapz(mean_over_roc, mean_fpr)
		aucs = [trapz(ROC[1], ROC[0]) for ROC in spec_rocs]
		std_auc = np.std(aucs)

		over_fig, over_ax = plt.subplots()
		over_ax.plot(mean_fpr, mean_over_roc, color ='C0', label=f'Mean ROC, AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
		for ROC in over_rocs:
			over_ax.plot(ROC[0], ROC[1], alpha=0.2)
		over_ax.fill_between(mean_fpr, over_bot, over_top, color='C0', alpha=.1, label=f'$\pm$ 1 std')
		over_ax.grid(alpha=0.2)
		over_ax.legend()
		over_ax.set_xlabel('False Positive Rate')
		over_ax.set_ylabel('True Positive Rate')
		over_ax.title.set_text("Overall Survival Model")
		plt.savefig('Overall_Mean_ROC.png', dpi = 230)

		both_fig, both_ax = plt.subplots()
		both_ax.plot(mean_fpr, mean_spec_roc, color ='C0', label=f'DSS AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
		both_ax.fill_between(mean_fpr, spec_bot, spec_top, color='C0', alpha=.1)#, label=f'$\pm$ 1 DSS std')
		both_ax.plot(mean_fpr, mean_over_roc, color ='C1', label=f'Overall AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
		both_ax.fill_between(mean_fpr, over_bot, over_top, color='C1', alpha=.1)# label=f'$\pm$ 1 OVR std')
		both_ax.grid(alpha=0.2)
		both_ax.legend(loc=4)
		both_ax.set_xlabel('False Positive Rate')
		both_ax.set_ylabel('True Positive Rate')
		both_ax.title.set_text("Receiver Operating Characteristic Curve")
		plt.savefig('Both_Mean_ROC.png', dpi = 230)




		# Calibration Plot
# Calibration Plot
		both_calfig, both_calax = plt.subplots()
		mean_spec_cal = []
		mean_over_cal = []
		for CAL in spec_cals:
   			 interp_spec_cal = interp(mean_fpr, CAL[1], CAL[0])
   			 mean_spec_cal.append(interp_spec_cal)

   		std_spec_cal = np.std(np.array(mean_spec_cal), axis=0)
		mean_spec_cal = np.mean(np.array(mean_specs_cal), axis=0)
		spec_top = np.minimum(mean_spec_cal + std_spec_cal, 1)
		spec_bot = np.maximum(mean_spec_cal - std_spec_cal, 0)
		spec_mbs = [np.polyfit(CAL[1], CAL[0], 1) for CAL in spec_cals]
		spec_ms = [m[0] for m in spec_mbs]
		spec_bs = [b[1] for b in spec_mbs]
		spec_m_std = np.std(spec_ms)
		spec_b_std = np.std(spec_bs)
		spec_m_mean = np.mean(spec_ms)
		spec_b_mean = np.mean(spec_bs)

		for CAL in over_cals:
   			 interp_over_cal = interp(mean_fpr, CAL[1], CAL[0])
   			 mean_over_cal.append(interp_over_cal)

   		std_over_cal = np.std(np.array(mean_over_cal), axis=0)
		mean_over_cal = np.mean(np.array(mean_overs_cal), axis=0)
		over_top = np.minimum(mean_over_cal + std_over_cal, 1)
		over_bot = np.maximum(mean_over_cal - std_over_cal, 0)
		over_mbs = [np.polyfit(CAL[1], CAL[0], 1) for CAL in over_cals]
		over_ms = [m[0] for m in over_mbs]
		over_bs = [b[1] for b in over_mbs]
		over_m_std = np.std(over_ms)
		over_b_std = np.std(over_bs)
		over_m_mean = np.mean(over_ms)
		over_b_mean = np.mean(over_bs)

		both_calax.plot(mean_fpr, mean_spec_cal)
		both_calax.fill_between(mean_fpr, spec_bot, spec_top, legend = 'Specific', color='C0', alpha=.1)#, label=f'$\pm$ 1 DSS std')
		both_calax.plot(mean_fpr, mean_over_cal)
		both_calax.fill_between(mean_fpr, over_bot, over_top, legend = 'Overall', color='C0', alpha=.1)#, label=f'$\pm$ 1 DSS std')
		both_calax.plot([0, 1], [0, 1], 'r--')
		both_calax.grid(alpha=0.2)
		both_calax.legend(loc=4)
		both_calax.set_xlabel('Predicted Probability')
		both_calax.set_ylabel('Observed Population')
		both_calax.title.set_text("Calibration Curve")
		plt.savefig("BothCalibration.png", dpi=230)


	print(f'\n\nGenerated Specific vs Overall ROC plot for test set #{k} using lastest models in ./ROCs/\n\n')

	return spec_rocs, over_rocs, spec_cals, over_cals






