# Uncertainty Quantification 
Please read the following to have some information about the data generation process:

1. The model is picked from this web page (https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html). Please refer 
   to Eqs. 19-F & 26-F. This page also has some references at the end (including books and papers) where you will find furthur            information about this cartpole model.

2. There are three Input forcing functions. Two of those are commented out. The resulting time-series will look different depending on 
   force input function. 

3. The columns in the csv file contains the following information in the same order:
   theta, theta_d, x, x_d, forceIn, theta_dd, x_dd
   The first four columns are input features, but I am skipping 'x' since it does not show up anywhere in cartpole final equations.      So, we have four input features including forceIn and last two columns are desired outputs.  

Markup: 4. At first, we generate deterministic data (no randomness)

           4.1. The code to generate data using this deterministic model is in the following file: Code/dataGeneration.ipynb
           4.2. The data in the form of csv file is in the following directory: deterministicCase
   
Markup: 5. Noise will be added to the cartpole model and model becomes stochastic 
   
           5.1. The code to generate data using this stochastic model is in the following file: Code/noisyDataGeneration.ipynb
           5.2. The data in the form of csv file is in the following directory: noisyCase
           5.3. Initially, we are adding additive Gaussian noise to the mu_p (co-efficient of friction for pole) 
           5.4. Later on, we can perturb other parameters as well, like mu_c, etc. 
