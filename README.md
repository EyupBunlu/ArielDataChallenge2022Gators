2022 NeurIPS Ariel Data Challenge

Gator's First Place Regular Track and Second Place Light Track Solution

Abstract: We present the solution to the Ariel Data Challenge of the “Gators” team from the Physics Department at the University of Florida. We train a model of interconnected neural networks to estimate a posterior distribution over possible exoplanetary atmospheric chemical compositions and surface temperatures from their transit spectrum and system auxiliary priors. A significant improvement of the model’s prediction was made by preprocessing the data using physically motivated feature engineering. The constructed model consists of several fully connected neural networks which use concatenations or products of the outputs of previous modules as inputs. To minimize the Wasserstein-2 distance while reducing the complexity of our model, we trained on a parameterization of the estimate of the posterior distribution. In cases when a concentration is too small to be detected, a functional term is added to reproduce the observed effect of the prior where the posterior distribution ends up being uniform across all compatible concentrations. 


 
