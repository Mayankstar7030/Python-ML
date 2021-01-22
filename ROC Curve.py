#!/usr/bin/env python
# coding: utf-8

# In[1]:



#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[2]:


#Initializing counters
TP=0
TN=0
FP=0
FN=0


# In[3]:


#Reading the file
diabetesDF = pd.read_csv('diabetes.csv')


# In[5]:


#Training data
dfTrain = diabetesDF[:500]


# In[6]:


#Testing Data
dfTest = diabetesDF[500:]


# In[7]:


#Data Manipulations
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))


# In[8]:


#Normalizing
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0
            
trainData = (trainData - means)/stds
testData = (testData - means)/stds


# In[9]:


diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)


# In[10]:


prediction=diabetesCheck.predict(testData)


# In[11]:


for i in range(0,268):
    if testLabel[i]==1 and prediction[i]==1:
        TP=TP+1
    if testLabel[i]==0 and prediction[i]==0:
        TN=TN+1
    if testLabel[i]==0 and prediction[i]==1:
        FP=FP+1        
    if testLabel[i]==1 and prediction[i]==0:
        FN=FN+1
        

 


# In[12]:


#accuracy
Acc=(TP+TN)/268
Acc


# In[13]:


accuracyModel = diabetesCheck.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")


# In[14]:


#Precision
Prec=TP/(TP+FP)
Prec


# In[15]:


#Recall
Rec=TP/(TP+FN)
Rec


# In[30]:


# predict
predictionProbability = diabetesCheck.predict_proba(testData)[:,1]
predictionProbability


# In[21]:


print(classification_report(testLabel, prediction))
from sklearn.metrics import classification_report, confusion_matrix


# In[31]:


fpr, tpr, thresholds = roc_curve(testLabel, predictionProbability)


# In[32]:


plt.plot(fpr, tpr)


# In[ ]:





# In[ ]:





# In[ ]:




