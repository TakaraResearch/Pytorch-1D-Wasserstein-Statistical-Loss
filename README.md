## 1D WASSERSTEIN STATISTICAL DISTANCES LOSSES IN PYTORCH

&nbsp;

### Introduction:
This repository is created to provide a **Pytorch** wasserstein statistical loss solution for a pair of 1D weight distributions.
 
&nbsp;
 
### How To:
All core functions of this repository are created in **pytorch_stats_loss.py**. To introduce the related Pytorch losses, just add this file into your project and import it at your wish.  
A group of dependent examples of related functionalities could be found in **stats_loss_testing_file.py**.  
**Pytorch_Statistical_Losses_Combined.py** makes a combination of the loss functions and their examples, and provides a "one click and run" program for the convinence of interested users.  
**pytorch_stats_loss.py** should be regarded as the center file of this project. 
 
&nbsp;
 
### Points of Background Information:
Statistial Distances for 1D weight distributions  
Inspired by Scipy.Stats Statistial Distances for 1D distributions  
**Pytorch Version, supporting Autograd to make a valid Loss for deep learning**  
Supposing Inputs are Groups of Same-Length Weight Vectors  
Instead of (Points, Weight), full-length Weight Vectors are taken as Inputs  
**Losses are built up based on the result of CDF calculations**
 
&nbsp;
 
### If you want to know more:
Check Scipy.Stats module for more background knowledge.  
Check Pytorch to know more about deep learning.

