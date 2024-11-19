# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:45:26 2024

@author: lmackin_lbs
"""

import pandas as pd
import numpy as np
import xlrd
import statsmodels.api as sm
from scipy import stats
from statsmodels.sandbox.regression.gmm import IV2SLS
import linearmodels.iv.model as lm


consumption_data = pd.read_excel("PS4Data.xls")

consumption_data["consumption"] = (consumption_data["real consumption of nondurables"] + consumption_data["real consumption of services"]) / consumption_data["population"]

#Create lagged variables

consumption_data["c_lag1"] = consumption_data["consumption"].shift(1)
consumption_data["c_lag2"] = consumption_data["consumption"].shift(2)
consumption_data["c_lag3"] = consumption_data["consumption"].shift(3)
consumption_data["c_lag4"] = consumption_data["consumption"].shift(4)
consumption_data["c_lag5"] = consumption_data["consumption"].shift(5)
consumption_data["c_lag6"] = consumption_data["consumption"].shift(6)
consumption_data["income"] = consumption_data["real disposable income"] / consumption_data["population"]
consumption_data["y_lag1"] = consumption_data["income"].shift(1)

part_a_data = consumption_data.drop(['c_lag5', 'c_lag6', 'y_lag1'], axis = 1)

part_a_data = part_a_data.dropna()

## PART A ##

x = part_a_data[["c_lag1", "c_lag2", "c_lag3", "c_lag4"]]
#add constant
x = sm.add_constant(x)
y = part_a_data[["consumption"]]

model = sm.OLS(y,x)
results = model.fit()
results_summary = results.summary()

#Jointly test if coefficients on c_lag2 - clag4 are different from 0
#Let's test this in two ways to validate results.
hypotheses = '(c_lag2 = 0), (c_lag3 = 0), (c_lag4 = 0)'
f_test2 = results.f_test(hypotheses)

hypotheses2 = '(c_lag2 = c_lag3 = c_lag4 = 0)'
f_test = results.f_test(hypotheses2)

#Calculate the critical value
sig_level = 0.05
f_crit = stats.f.ppf(1 - sig_level, f_test.df_num, f_test.df_denom)

## PART C ##

#Let's create a separate data for our IV regression
IV_data = consumption_data[["consumption", "income", "y_lag1", "c_lag1", "c_lag2", "c_lag3", "c_lag4", "c_lag5", "c_lag6"]]
#Here, we drop Nan values
IV_data = IV_data.dropna()

#Create variables
IV_data["Ct/Ct-1"] = np.log(IV_data["consumption"] / IV_data["c_lag1"])
IV_data["Yt/Yt-1"] = np.log(IV_data["income"] / IV_data["y_lag1"])

#Create instruments
IV_data["Ct-2/Ct-3"] = np.log(IV_data["c_lag2"] / IV_data["c_lag3"])
IV_data["Ct-3/Ct-4"] = np.log(IV_data["c_lag3"] / IV_data["c_lag4"])
IV_data["Ct-4/Ct-5"] = np.log(IV_data["c_lag4"] / IV_data["c_lag5"])
IV_data["Ct-5/Ct-6"] = np.log(IV_data["c_lag5"] / IV_data["c_lag6"])

#In this section, we're going to obtain our results using two different methods to
#validate our results.

#Now, we designate our exogenous and instrumental variables
exog = IV_data["Yt/Yt-1"]
exog = sm.add_constant(exog)

instruments = IV_data[["Ct-2/Ct-3", "Ct-3/Ct-4", "Ct-4/Ct-5", "Ct-5/Ct-6"]]
instruments = sm.add_constant(instruments)

endog = IV_data["Ct/Ct-1"]

iv2sls_model = IV2SLS(endog=endog, exog=exog, instrument=instruments)
iv2sls_model_results = iv2sls_model.fit()

#Print the training summary
print(iv2sls_model_results.summary())


#Let's run a different way and see if we get similar results
iv_data_2 = sm.add_constant(data = IV_data, prepend = False)

second_2sls_model = lm.IV2SLS(dependent = iv_data_2[["Ct/Ct-1"]], exog = iv_data_2[["const"]],
                              endog = iv_data_2[["Yt/Yt-1"]],
                              instruments = iv_data_2[["Ct-2/Ct-3", "Ct-3/Ct-4", "Ct-4/Ct-5", "Ct-5/Ct-6"]]
                              ).fit(cov_type = "robust")

print(second_2sls_model)

#The models match! We will use the second 2SLS model for the remainder of the problem

#f = open('2SLS_reg.tex', 'w')
#f.write(second_2sls_model.summary.as_latex())
#f.close()

#First, let's run tests and assume conditional homoskedasticity.
#This is assumed when computing Wu_Hausman and Sargan statistics

#Perform tests to see if endogenous variable is exogenous
wu_hausman = second_2sls_model.wu_hausman()
print(wu_hausman)
f_crit2 = stats.f.ppf(1 - 0.05, 1, 211)
print(f_crit2)

#Sargan test of overidentification
sargan_test = second_2sls_model.sargan
print(sargan_test)
#Calculate critical value
sargan_crit = stats.chi2.ppf(1 - 0.05, 3)
print(f"Critical value : {sargan_crit:.4f}")



#Note: Now, let's compute tests without assuming conditional homoskedasticity.
# We can use two of Wooldridge's tests, as they are robust to heteroskedasticity
#Wooldridge's test for exogeneity

wooldridge = second_2sls_model.wooldridge_score
print(wooldridge)
wooldridge_exog_crit = stats.chi2.ppf(1 - 0.05, 1)
print(wooldridge_exog_crit)

#wooldridge_reg = second_2sls_model.wooldridge_regression
#print(wooldridge_reg)

wooldridge_overid_test = second_2sls_model.wooldridge_overid
print(wooldridge_overid_test)

wooldridge_overid_crit = stats.chi2.ppf(1 - 0.05, 3)
print(wooldridge_overid_crit)

