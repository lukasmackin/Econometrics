# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:30:37 2024

@author: lmackin_lbs
"""

import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
import scipy.stats

#Quick path check to figure out where I'm working
path = os.getcwd()
#print(path)

titanic_data = pd.read_csv("TitanicData.csv")

#Remove NA observations in specified columns
titanic_data.dropna(subset = ['Survived', 'Pclass', 'Sex', 'Age'], inplace = True)

df_for_probit = titanic_data[['Survived', 'Pclass', 'Sex', 'Age']]
df_for_probit['Sex'].replace(['female','male'], [0,1], inplace=True)


#define our X and Y variable for probit
y = df_for_probit[["Survived"]]
X = df_for_probit[['Pclass', 'Sex', 'Age']]
X = sm.add_constant(X)

#run probit
probit_model=sm.Probit(y,X)
result=probit_model.fit()
print(result.summary2())
result_summary = result.summary2()

#Output these results to a .tex file
#f = open('probit_reg.tex', 'w')
#f.write(result_summary.as_latex())
#f.close()

#Part B: Calculating the probability
prob_survival = scipy.stats.norm.cdf(-1.8928)

#Part C: Calculating effect
change_in_survival_prob = scipy.stats.norm.cdf(-0.4516) - scipy.stats.norm.cdf(-1.8928)


#PART D: CALCULATION OF GRADIENT COMPONENTS
#First, define phat_1 and phat_3
phat_1 = result.params.iloc[0] + result.params.iloc[1] + result.params.iloc[2] + result.params.iloc[3]*55
phat_3 = result.params.iloc[0] + result.params.iloc[1]*3 + result.params.iloc[2] + result.params.iloc[3]*55

#beta_1 derivative
dbeta_1 = scipy.stats.norm.pdf(phat_1) - scipy.stats.norm.pdf(phat_3)

#beta_2 derivative
dbeta_2 = scipy.stats.norm.pdf(phat_1) - 3*scipy.stats.norm.pdf(phat_3)

#beta_3 derivative
dbeta_3 = scipy.stats.norm.pdf(phat_1) - scipy.stats.norm.pdf(phat_3)

#beta_4 derivative
dbeta_4 = 55*scipy.stats.norm.pdf(phat_1) - 55*scipy.stats.norm.pdf(phat_3)

#Now, construct the gradient vector
p_gradient = np.array([[dbeta_1, dbeta_2, dbeta_3, dbeta_4]])
var_covar_array = np.asarray(result.cov_params())

#Calculate the variance term
p_variance = p_gradient @ var_covar_array @ np.transpose(p_gradient)

#To get the standard error, we take the square root of this
standard_error = np.sqrt(p_variance[0])