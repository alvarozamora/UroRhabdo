import torch
import pdb
import copy
import glob
import numpy as np
import torch.nn as nn
from LSUV import LSUVinit
#from model import *
#from utils import *
from pytorch2keras import pytorch_to_keras
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model as KModel
import tensorflow.keras.backend as KB

print("Verifying Keras Models")



final_specs = nn.ModuleList([torch.load(model) for model in glob.glob("Final_DSS_*.pth")])
final_overs = nn.ModuleList([torch.load(model) for model in glob.glob("Final_OS_*.pth")])

def Spec(x):
	output = torch.mean(torch.stack([final_specs[i](x) for i in range(len(final_specs))], dim=0), dim=0)
	return output


def Over(x):
	output = torch.mean(torch.stack([final_overs[i](x) for i in range(len(final_overs))], dim=0), dim=0)
	return output

k_spec = tf.keras.models.load_model('Final_DSS.h5')
k_over = tf.keras.models.load_model('Final_OS.h5')


Data = np.load("Data.npz")

N = 5
N = np.random.choice(range(len(Data['Data'])), N)
print('Verifying IDs:', Data['IDs'][N])

keras_spec_preds = k_spec.predict(Data['Data'][N])
keras_over_preds = k_over.predict(Data['Data'][N])
#pdb.set_trace()
torch_spec_preds = Spec(torch.Tensor(Data['Data'])[N].float())
torch_over_preds = Over(torch.Tensor(Data['Data'])[N].float())


print('Keras Predictions: DSS, OS = ', keras_spec_preds, keras_over_preds)
print('Torch Predictions: DSS, OS = ', torch_spec_preds[:,0].data.numpy(), torch_over_preds[:,0].data.numpy())

#Data, spec, over, IDs = torch.Tensor(Data['Data']), torch.Tensor(Data['spec']), torch.Tensor(Data['over']), Data['IDs']



