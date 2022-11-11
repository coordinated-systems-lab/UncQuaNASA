# Uncertainty Quantification 
## Data
Please read the following to have some information about the data generation process:

1. The model is picked from this web page (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html). Please refer 
   to Eqs. 19-F & 26-F. This page also has some references at the end (including books and papers) where you will find furthur            information about this cartpole model.

2. There are three Input forcing functions. Two of those are commented out. The resulting time-series will look different depending on 
   force input function. 

3. The columns in the csv file contains the following information in the same order:
   theta, theta_d, x, x_d, forceIn, theta_dd, x_dd
   The first four columns are input features, but I am skipping 'x' since it does not show up anywhere in cartpole final equations.      So, we have four input features including forceIn and last two columns are desired outputs.  

4. At first, we generate deterministic data (no randomness)

   1. The code to generate data using this deterministic model is in the following file: Data/Code/dataGeneration.ipynb
   2. The data in the form of csv file is in the following directory: Data/deterministicCase
   
5. Noise will be added to the cartpole model and model becomes stochastic 
   
   1. The code to generate data using this stochastic model is in the following file: Data/Code/noisyDataGeneration.ipynb
   2. The data in the form of csv file is in the following directory: Data/noisyCase
   3. Initially, we are adding additive Gaussian noise to the mu_p (co-efficient of friction for pole) 
   4. Later on, we can perturb other parameters as well, like mu_c, etc. 

## Models
Some information about Models directory is given below:

1. This directory contains code for different models we have tried so far. These include: SINDYc, Gaussian Processes (GP). 

2. Model learned using different methods are saved in Models/learnedModels. Within that, there are multiple folders coresponding to    different models

3. SINDYc did not give us good result and there will no saved learned models for it. 

4. GP was implemented with GPy and GPyTorch. Both libraries gave good results with GPyTorch being really fast because it uses GPU.      But the resuls of GPy are more reliable. With GPyTorch, you have to do a lot of manual tuning to get good results. The learned      models are saved.

## Results 
Information about resuls  directory 

1. Plots of some of the results generated are given in this directory. There are different sub-directories corresponding to            different models. 
