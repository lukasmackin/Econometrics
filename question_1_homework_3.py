# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:47:35 2024

@author: lmackin_lbs
"""

import pandas as pd
import numpy as np

sp_500_data = pd.read_excel("SP500Index.xlsx")

sp_500_data = sp_500_data.drop(["Date of Observation"], axis = 1)

x_t = np.zeros(len(sp_500_data))

for i in range (1, len(sp_500_data)):
    x_t[i] = np.log(sp_500_data.iloc[i] / sp_500_data.iloc[0])
    
    
#First, compute delta_MLE

delta_MLE = x_t[701] / 701

#Now, compute sigma_hat_mle

sigma_vector = np.zeros(len(sp_500_data))
for i in range (1, len(sp_500_data)):
    sigma_vector[i] = (x_t[i] - x_t[i-1] - delta_MLE)**2
    
    
#Get sigma_hat_mle
sigma_hat_MLE = np.sum(sigma_vector) / (len(sp_500_data) - 1)

#take square root of sigma_hat_mle
sqrt_sigma_hat_mle = np.sqrt(sigma_hat_MLE)



#QUICK P-VALUE CALCULATION FOR QUESTION 3
from scipy import stats

p_value = 1 - stats.chi2.cdf(0.5, 1)
    