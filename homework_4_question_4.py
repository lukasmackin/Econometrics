# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:06:49 2024

@author: lmackin_lbs
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("minwage.txt", delimiter="\t", na_values = ".")
df = df.astype('float')

#Create variables fte and fte2
df["fte"] = df["empft"] + df["nmgrs"] + df["emppt"]/2
df["fte2"] = df["empft2"] + df["nmgrs2"] + df["emppt2"]/2

#Drop NA values of the desired observations
df = df.dropna(subset = ["wagest", "wagest2", "empft", "empft2",
                         "nmgrs", "nmgrs2", "emppt", "emppt2"])
df = df.reset_index(drop = True)

# Part A
#Let's fill out the table 
#Create a Pennsylvannia table
pa_df = df[df["state"] == 0]
nj_df = df[df["state"] == 1]


#Get PA before and after means
#Not sure if this should be fte or wagest
pa_before_mean = np.mean(pa_df["fte"])
pa_after_mean = np.mean(pa_df["fte2"])

#Get NJ before and after means
nj_before_mean = np.mean(nj_df["fte"])
nj_after_mean = np.mean(nj_df["fte2"])

#Compute diffs
pa_diff = pa_before_mean - pa_after_mean
nj_diff = nj_before_mean - nj_after_mean
pa_nj_before = pa_before_mean - nj_before_mean
pa_nj_after = pa_after_mean - nj_after_mean

#I use these to manually fill in table



## Part b ##
#To create out data, let's create two subset dataframes and concatenate

#Time 0 dataframe

time_0_df = df[["fte", "state"]]
#Add time = 0 column
time_0_df['time'] = pd.Series([0 for x in range(len(time_0_df))]) 
time_0_df = time_0_df.rename(columns = {"fte":"employment"})


#Time 1 dataframe

time_1_df = df[["fte2", "state"]]
#Add time = 1 column
time_1_df['time'] = pd.Series([1 for x in range(len(time_1_df))]) 
time_1_df = time_1_df.rename(columns = {"fte2":"employment"})

#Concatenate the dataframes
diff_diff_df = pd.concat([time_0_df, time_1_df])

#Add the dummy variable column
diff_diff_df["dummy"] = diff_diff_df["state"] * diff_diff_df["time"]

#Run the regression
y = diff_diff_df["employment"]
x = diff_diff_df[["state", "time", "dummy"]]
x = sm.add_constant(x)

model = sm.OLS(y,x)
results = model.fit()
results_summary = results.summary()
print(results.summary())

#f = open('hw_4_question_4b_reg.tex', 'w')
#f.write(results_summary.as_latex())
#f.close()

  
## Part c ##
#We basically redo part b excpet we also take chain as a variable. We could code
#this more efficiently by combining all variables initially. But for now,
#I settle for suboptimality

#Time 0 dataframe: lead with c_ to denote this dataframe is for part c

c_time_0_df = df[["fte", "state", "chain"]]
#Add time = 0 column
c_time_0_df['time'] = pd.Series([0 for x in range(len(c_time_0_df))]) 
c_time_0_df = c_time_0_df.rename(columns = {"fte":"employment"})

#Create dummy variables for chain. 
c_time_0_df = pd.get_dummies(c_time_0_df, columns=['chain'], prefix='chain', drop_first=True, dtype = float)

#Time 1 dataframe

c_time_1_df = df[["fte2", "state", "chain"]]
#Add time = 1 column
c_time_1_df['time'] = pd.Series([1 for x in range(len(c_time_1_df))]) 
c_time_1_df = c_time_1_df.rename(columns = {"fte2":"employment"})

#Create dummy variables for chain. 
c_time_1_df = pd.get_dummies(c_time_1_df, columns=['chain'], prefix='chain', drop_first=True, dtype = float)

#Concatenate the dataframes
c_diff_diff_df = pd.concat([c_time_0_df, c_time_1_df])

#Add the dummy variable column
c_diff_diff_df["dummy"] = c_diff_diff_df["state"] * c_diff_diff_df["time"]

#Run the regression
y_c = c_diff_diff_df["employment"]
x_c = c_diff_diff_df[["state", "time", "chain_2.0", "chain_3.0", "chain_4.0", "dummy"]]
x_c = sm.add_constant(x_c)

model_c = sm.OLS(y_c,x_c)
results_c = model_c.fit()
results_c_summary = results_c.summary()
print(results_c.summary())

#f = open('hw_4_question_4c_reg.tex', 'w')
#f.write(results_c_summary.as_latex())
#f.close()

## Part D ##
#Create separate Gap dataframe
gap_df = df
#Create dependent variable employment_change
gap_df['employment_change'] = gap_df['fte2'] - gap_df['fte']
#Create gap variable
#gap_df['gap'] = [0 if x == 0 or y >= 5.05 else 1 for x,y in gap_df['state'], gap_df['wagest']]

gap_df['gap'] = np.where(((gap_df['state']==0) | (gap_df['wagest'] >= 5.05)), 0, (5.05 - gap_df['wagest'])/gap_df['wagest'])

#Create variables for regression
x_gap = gap_df[['gap']]
x_gap = sm.add_constant(x_gap)
y_gap = gap_df[['employment_change']]

#Run regression
model_d = sm.OLS(y_gap, x_gap)
results_d = model_d.fit()
print(results_d.summary())

#obtain mean of gap for new jersey restaurants
#nj_gap = gap_df[['state']==0]
nj_gap = gap_df.loc[gap_df['state'] == 1]
nj_mean_gap = np.mean(nj_gap['gap'])

#Multiply by the regression parameter
diff_diff_estimate = results_d.params['gap'] * nj_mean_gap
print(diff_diff_estimate)
#Not that similar because we didn't control for other factors

## Part E ##
#First, create dummy variables for chain
gap_df = pd.get_dummies(gap_df, columns=['chain'], prefix='chain', drop_first=True, dtype = float)


#x_e = gap_df[['gap', 'state', 'chain', 'own']]

x_e = gap_df[['gap', 'state', "chain_2.0", "chain_3.0", "chain_4.0", 'own']]
x_e = sm.add_constant(x_e)
y_e = gap_df[['employment_change']]

#Run regression
model_e = sm.OLS(y_e, x_e)
results_e = model_e.fit()
print(results_e.summary())

#Test whether coefficient is zero
hypothesis = 'state = 0'
t_test = results_e.t_test(hypothesis)
print(t_test)

#Fail to reject the null hypothesis that the coefficient is 0