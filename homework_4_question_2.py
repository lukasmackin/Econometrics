# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:44:39 2024

@author: lmackin_lbs
"""

import pandas as pd
import numpy as np
from statsmodels.sandbox.regression.gmm import GMM
from scipy.stats import t


data = pd.read_excel('ccapm.xlsx',header=None, names=['cratio', 'rrate', 'e'])

# Define the model and moment conditions
class CCAPMGMM(GMM):
    def momcond(self, params):
        beta, gamma = params
        cratio = self.endog[:, 0]
        rrate = self.endog[:, 1]
        cratiolag = self.exog[:, 0]
        rratelag = self.exog[:, 1]
        elag = self.exog[:, 2]
        
        m1 = beta * (cratio ** -gamma) * rrate - 1
        m2 = m1 * cratiolag
        m3 = m1 * rratelag
        m4 = m1 * elag

        return np.column_stack((m1, m2, m3, m4))

# Extract data columns for use in GMM
cratio = data['cratio'].values
rrate = data['rrate'].values
e = data['e'].values

# Set up the GMM model
initial_params = np.array([1.0, 1.0])  # Starting values for beta and gamma
instruments = np.column_stack((cratio[:-1], rrate[:-1], e[:-1], np.ones(len(cratio) - 1)))  # Instruments for the GMM estimation

# Correct the GMM setup with instruments
model = CCAPMGMM(endog=np.column_stack((cratio[1:],rrate[1:])), exog=instruments, instrument=instruments)

# Estimate parameters
results = model.fit(start_params=initial_params, maxiter=100)
results_summary = results.summary()
print(results.summary())

#f = open('hw_4_question_2a_reg.tex', 'w')
#f.write(results_summary.as_latex())
#f.close()

## Part B ##
results_b = model.fit(start_params=initial_params, maxiter=1000,weights_method="hac",wargs={"maxlag": 5})
results_b_summary = results_b.summary()

#f = open('hw_4_question_2b_reg.tex', 'w')
#f.write(results_b_summary.as_latex())
#f.close()

## Part C ##

# Hypothesis testing for beta = 0.98
beta_estimate = results_b.params[0]
beta_se = results_b.bse[0]
beta_null = 0.98

# Compute t-statistic
t_stat = (beta_estimate - beta_null) / beta_se

# Degrees of freedom (large sample, use normal approximation)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=len(cratio) - 1))

# Test result
hypothesis_test_result = {
    "Null Hypothesis": "Beta = 0.98",
    "Beta Estimate": beta_estimate,
    "Standard Error": beta_se,
    "t-Statistic": t_stat,
    "p-Value": p_value,
    "Reject Null (95%)": p_value < 0.05
}
hypothesis_test_result

