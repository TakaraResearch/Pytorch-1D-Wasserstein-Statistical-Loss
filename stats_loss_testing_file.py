#############################################################
#  Testing Script for Statistical Loss Function in Pytorch  #
#############################################################
#-----------------------------------------------------------#
# This is a testing script for pytorch statistical loss function
# pytorch_stats_loss.py should be in the same folder with this testing script

import numpy as np
from scipy import stats
import torch
from torch.autograd import Variable
import pytorch_stats_loss as stats_loss
# pytorch_stats_loss.py is the script for loss definitions

# Testing Script for Statistical Loss Function in Pytorch

BATCH = 16
DIM = 32

# Making Data for Testing
vec_1 = np.random.random((BATCH,DIM))
vec_2 = np.random.random((BATCH,DIM))
vec_list = np.arange(DIM)

# Making Scipy Numpy Results
result_1=0
result_2=0
for i in range(BATCH):
    vec_dist_1 = stats.wasserstein_distance(vec_list, vec_list, vec_1[i], vec_2[i])
    vec_dist_2 = stats.energy_distance(vec_list,vec_list,vec_1[i],vec_2[i])
    result_1 += vec_dist_1
    result_2 += vec_dist_2
print("Numpy-Based Scipy Results: \n",
      "Wasserstein distance",result_1/BATCH,"\n",
      "Energy distance",result_2/BATCH,"\n")

# Making Pytorch Variable Calculations
tensor_1=Variable(torch.from_numpy(vec_1))
tensor_2=Variable(torch.from_numpy(vec_2),requires_grad=True)
tensor_3=Variable(torch.rand(BATCH+1,DIM))

# Show results
print("Pytorch-Based Results:")
print("Wasserstein loss",stats_loss.torch_wasserstein_loss(tensor_1,tensor_2).data,stats_loss.torch_wasserstein_loss(tensor_1,tensor_2).requires_grad)
print("Energy loss",stats_loss.torch_energy_loss(tensor_1,tensor_2).data,stats_loss.torch_wasserstein_loss(tensor_1,tensor_2).requires_grad)
print("p == 1.5 CDF loss", stats_loss.torch_cdf_loss(tensor_1,tensor_2,p=1.5).data)
print("Validate Checking Errors:", stats_loss.torch_validate_distibution(tensor_1,tensor_2))
#torch_validate_distibution(tensor_1,tensor_3)