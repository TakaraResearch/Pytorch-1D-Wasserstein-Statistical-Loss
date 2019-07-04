import numpy as np
from scipy import stats
import torch
from torch.autograd import Variable

#Testing Script for Statistical Loss Function

BATCH = 4
DIM = 6

vec_1 = np.random.random((BATCH,DIM))
vec_2 = np.random.random((BATCH,DIM))
vec_list = np.arange(DIM)

# Making Scipy Results
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


tensor_1=Variable(torch.from_numpy(vec_1))
tensor_2=Variable(torch.from_numpy(vec_2),requires_grad=True)
tensor_3=Variable(torch.rand(BATCH+1,DIM))


#######################################################
#       STATISTICAL DISTANCES(LOSSES) IN PYTORCH      #
#######################################################

## Statistial Distances for 1D weight distributions
## Inspired by Scipy.Stats Statistial Distances for 1D
## Pytorch Version, supporting Autograd to make a valid Loss
## Supposing Inputs are Groups of Same-Length Weight Vectors
## Instead of (Points, Weight), full-length Weight Vectors are taken as Inputs
## Code Written by E.Bao, CASIA


def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))

def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss

def torch_validate_distibution(tensor_a,tensor_b):
    # Zero sized dimension is not supported by pytorch, we suppose there is no empty inputs
    # Weights should be non-negetive, and with a positive and finite sum
    # We suppose all conditions will be corrected by network training
    # We only check the match of the size here
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Input weight tensors must be of the same size")

print("Pytorch-Based Results:")
print("Wasserstein loss",torch_wasserstein_loss(tensor_1,tensor_2).data,torch_wasserstein_loss(tensor_1,tensor_2).requires_grad)
print("Energy loss",torch_energy_loss(tensor_1,tensor_2).data,torch_wasserstein_loss(tensor_1,tensor_2).requires_grad)
print("p == 1.5 CDF loss", torch_cdf_loss(tensor_1,tensor_2,p=1.5).data)
print("Validate Checking Errors:", torch_validate_distibution(tensor_1,tensor_2))
#torch_validate_distibution(tensor_1,tensor_3)


