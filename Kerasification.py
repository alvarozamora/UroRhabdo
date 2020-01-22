import torch
import pdb
import copy
import glob
import numpy as np
from LSUV import LSUVinit
#from model import *
#from utils import *
from pytorch2keras import pytorch_to_keras
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model as KModel
import tensorflow.keras.backend as KB


print("Keras-ifying Models")

final_specs = [torch.load(model) for model in glob.glob("Final_DSS_*.pth")]
final_overs = [torch.load(model) for model in glob.glob("Final_OS_*.pth")]

dummy_input = torch.randn(2,9)

k_spec = []
k_over = []

for i in range(len(final_specs)):
	final_specs[i].eval()
	spec_model_k = pytorch_to_keras(final_specs[i], dummy_input, verbose=True)
	k_spec.append(spec_model_k)
	spec_model_k.save(f'Final_DSS_{i:03d}.h5')
for i in range(len(final_overs)):
	final_overs[i].eval()
	over_model_k = pytorch_to_keras(final_overs[i], dummy_input, verbose=True)
	k_over.append(over_model_k)
	over_model_k.save(f'Final_OS_{i:03d}.h5')

pdb.set_trace()
spec_input = Input(shape=(9,))
spec_output = KB.mean(KB.concatenate([spec(spec_input) for spec in k_spec], axis=1), axis=1)

spec_combined_model_k = KModel(inputs=spec_input, outputs=spec_output)
spec_combined_model_k.save(f'Final_DSS.h5')

over_input = Input(shape=(9,))
over_output = KB.mean(KB.concatenate([over(over_input) for over in k_over], axis=1), axis=1)

over_combined_model_k = KModel(inputs=over_input, outputs=over_output)
over_combined_model_k.save(f'Final_OS.h5')

#spec_model_k = pytorch_to_keras(final_spec, spec_xtrain, verbose=True)#, (9,), verbose=True, names='short')  
#final_over = CombinedModel(over_models)
#orch.save(final_over, 'Combined_Overall.pth')