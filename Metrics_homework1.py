# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:38:35 2024

@author: lmackin_lbs
"""

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
#from statsmodels.formula.api import ols
import statsmodels.api as sm

#Quick path check to figure out where I'm working
path = os.getcwd()
#print(path)

data = pd.read_csv("ps1small.csv")

#Transform the variables we need to
data['lwage'] = np.log(data['wage'])
data = data.drop(['wage'], axis = 1)


experiment_df = pd.pivot(data, values = ['lwage'], columns=['age', 'education'])
experiment_df2 = experiment_df
#experiment_df2.columns = ["age"+str(s2)+"edu"+str(s3) for (s1, s2, s3) in experiment_df.columns.tolist()]
experiment_df2.columns = ["edu"+str(s3) + "age"+str(s2) for (s1, s2, s3) in experiment_df.columns.tolist()]
dummy_df = experiment_df2.notnull().astype('int')
dummy_df = dummy_df.reindex(sorted(dummy_df.columns), axis = 1)
#Concatenate our dummy_df with lwage
lwage_df = data[['lwage']]
joined_df = pd.concat([lwage_df, dummy_df], axis = 1)
joined_df = joined_df.reset_index(drop = True)

#Now, we can run our regression
joined_df.loc[:, joined_df.columns != "lwage"]

x_df = joined_df.loc[:, joined_df.columns != "lwage"]
y = joined_df.loc[:, joined_df.columns == "lwage"]
print(type(x_df))

model = sm.OLS(y,x_df)
results = model.fit()
coeff_results = results.params
print("The coefficients can be found in the coeff_results variable. They can also be found below:")
print(coeff_results)
"The coefficients and their standard errors can be found in the results_summary variable. Alternatively, they can be found below:"
results_summary = results.summary()
print(results_summary)
#Output results to .tex file
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('myreg.tex', 'w')
f.write(beginningtex)
f.write(results_summary.as_latex())
f.write(endtex)
f.close()
unrestricted_rss = results.ssr


#We need to calculate the restricted model now
x_restricted_model_df = data.loc[:, data.columns != "lwage"]
#We also need to add an interaction term column
x_restricted_model_df["age_education"] = x_restricted_model_df["age"] * x_restricted_model_df["education"]

x_restricted_model_df = sm.add_constant(x_restricted_model_df)
restricted_model = sm.OLS(y, x_restricted_model_df)
restricted_results = restricted_model.fit()
restricted_rss = restricted_results.ssr

#We can compute out F-statistic
F_stat = ((restricted_rss - unrestricted_rss)/26) / (unrestricted_rss/868)

