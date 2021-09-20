#!/usr/bin/env python
# coding: utf-8

# In[1]:

#importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


# loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('dataset.csv')


# In[4]:


type(loan_dataset)


# In[5]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[6]:


# number of rows and columns
loan_dataset.shape


# In[7]:


# statistical measures
loan_dataset.describe()


# In[8]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[9]:


# dropping the missing values
loan_dataset = loan_dataset.dropna()


# In[10]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[11]:


# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[12]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[13]:


# Dependent column values
loan_dataset['Dependents'].value_counts()


# In[14]:


# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)


# In[15]:


# dependent values
loan_dataset['Dependents'].value_counts()


# In[16]:


# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


# In[17]:


# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)


# In[18]:


# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[19]:


loan_dataset.head()
loan_dataset.corr()

# In[20]:


# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']


# In[21]:


print(X)
print(Y)


# In[22]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# In[24]:


classifier = svm.SVC(kernel='linear')


# In[25]:


#training the support Vector Macine model
classifier.fit(X_train,Y_train)


# In[26]:


# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)


# In[27]:


print('Accuracy on training data : ', training_data_accuray)


# In[30]:


# accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)


# In[31]:


print('Accuracy on test data : ', test_data_accuray)


# In[ ]:




