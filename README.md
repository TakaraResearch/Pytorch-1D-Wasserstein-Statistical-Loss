# 1D WASSERSTEIN STATISTICAL DISTANCES LOSSES IN PYTORCH
#### Introduction:
This repository is created to provide a **Pytorch** loss solution of statistical (wasserstein) distance for a pair of 1D weight distributions.

#### How To:
All core functions of this repository are created in **pytorch_stats_loss.py**. To introduce the related Pytorch Losses, just add this file into your project.  
A group of dependent examples of related functionalities could be found in **stats_loss_testing_file.py**  
**Pytorch_Statistical_Losses_Combined.py** makes a combination of the loss functions and their examples, and provides a "one click and run" program for the convinence of interested users.  

**pytorch_stats_loss.py** should be regarded as the center file of this project. 

#### Background Information:
Statistial Distances for 1D weight distributions  
Inspired by Scipy.Stats Statistial Distances for 1D distributions  
Pytorch Version, supporting Autograd to make a valid Loss  
Supposing Inputs are Groups of Same-Length Weight Vectors  
Instead of (Points, Weight), full-length Weight Vectors are taken as Inputs  
**Losses built up based on the result of CDF calculations**  
