#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION - PROOF P-VALUE


# In[1]:


# ## IMPORT LIBRARY


#import the libararies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


# ## IMPORT DATA

#import the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset.head(10)



#show independent variable
X = dataset.iloc[:,:-1].values
#X



#show dependent variable
y = dataset.iloc[:, 4]
#y



#missing the data
dataset.isnull().sum()



#handle the categorical data with LabelEncoder abd OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()
#X



#avoid dummy variables trap, so eliminate one columns that it is into the dummy variables
X = X[:, 1:]
#X


# In[13]:

## EX-1 ##

## CALCULATE VARIANCE STEP BY STEP

#select high correlated column
X_frame = pd.DataFrame(X)
X_new = X_frame.iloc[:,2]

#calculate step by step variance (also i did sqrt so calculated standart deviation)
#i want to handle magnitude so i did sqrt then calculated standard deviation not variance actually
#because null hypothesis say that: correlation is zero
#alternative hypothesis say that: correlation is not zero
#we try to reject null hypothesis
#so we should correlation each steps and calculate CI and p-value
#also we should draw a graph for proof
import math
k=list()
j=list()
for i in range(0,48,1):
    ind = i + 1
    k.append(math.sqrt((((abs(X_new[ind]-X_new[i]))**2)/2)))
    j.append(math.sqrt(((abs(y[ind]-y[i]))**2)/2))

#transformation
k_frame = pd.DataFrame(k)
j_frame = pd.DataFrame(j)


## DRAW A GRAPH FOR DEEP UNDERSTANDABLE ##

#j_frame.describe()
#j_frame.describe()

#p-value=0.0
#corr=1.0
#they are very pararrelled so high correlated
#then we can reject null hypothesis
#they are covered
#so p-value is too small almost zero
#because p-value say that they are correlated
#if not exists correlation p-value is high
#otherwise correlation is low because exist high correlation
plt.figure(1)
sns.distplot(k_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkgreen","label":"independent variable"})

sns.distplot(j_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkred","label":"dependent variable"});
plt.show();




# In[23]:


## EX-2 ##

## CALCULATE VARIANCE STEP BY STEP

#select high correlated column
X_frame = pd.DataFrame(X)
X_new = X_frame.iloc[:,0]

#calculate step by step variance (also i did sqrt so calculated standart deviation)
#i want to handle magnitude so i did sqrt then calculated standard deviation not variance actually
#because null hypothesis say that: correlation is zero
#alternative hypothesis say that: correlation is not zero
#we try to reject null hypothesis
#so we should correlation each steps and calculate CI and p-value
#also we should draw a graph for proof
import math
k=list()
j=list()
for i in range(0,48,1):
    ind = i + 1
    k.append(math.sqrt((((abs(X_new[ind]-X_new[i]))**2)/2)))
    j.append(math.sqrt(((abs(y[ind]-y[i]))**2)/2))

#transformation
k_frame = pd.DataFrame(k)
j_frame = pd.DataFrame(j)


## DRAW A GRAPH FOR DEEP UNDERSTANDABLE ##

#j_frame.describe()
#j_frame.describe()

#p-value=0.953
#corr=0.1
#they are not pararrelled so low correlated
#then we cannot reject null hypothesis
#they are not covered
#so p-value is too big almost is one
#because p-value say that they are not correlated
#if not exists correlation p-value is high
#otherwise correlation is low because exist low correlation
plt.figure(2)
sns.distplot(k_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkgreen","label":"independent variable"})

sns.distplot(j_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkred","label":"dependent variable"});
plt.show();





# In[23]:


## EX-3 ##

## CALCULATE VARIANCE STEP BY STEP

#select high correlated column
X_frame = pd.DataFrame(X)
X_new = X_frame.iloc[:,4]

#calculate step by step variance (also i did sqrt so calculated standart deviation)
#i want to handle magnitude so i did sqrt then calculated standard deviation not variance actually
#because null hypothesis say that: correlation is zero
#alternative hypothesis say that: correlation is not zero
#we try to reject null hypothesis
#so we should correlation each steps and calculate CI and p-value
#also we should draw a graph for proof
import math
k=list()
j=list()
for i in range(0,48,1):
    ind = i + 1
    k.append(math.sqrt((((abs(X_new[ind]-X_new[i]))**2)/2)))
    j.append(math.sqrt(((abs(y[ind]-y[i]))**2)/2))

#transformation
k_frame = pd.DataFrame(k)
j_frame = pd.DataFrame(j)


## DRAW A GRAPH FOR DEEP UNDERSTANDABLE ##

#j_frame.describe()
#j_frame.describe()

#p-value=0.123
#corr=0.7
#they are not pararrelled so high correlated
#then we can reject null hypothesis
#they are covered
#so p-value is big but not better than ex-1
#because p-value say that they are correlated
#if not exists correlation p-value is high
#otherwise correlation is low because exist high correlation
plt.figure(3)
sns.distplot(k_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkgreen","label":"independent variable"})

sns.distplot(j_frame, hist=False, 
             kde_kws={"alpha":1,"color":"darkred","label":"dependent variable"});
plt.show();






