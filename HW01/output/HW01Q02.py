
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


col_names = []
for i in range(60):
    col_names.append('x%d'%i)
col_names.append('label')


# In[3]:


#I shuffled the dataset once and kept it the same for all of the questions.
#The next three commented codes were executed once.
#data = pd.read_csv('F://Uni/992/Neural Networks/Homeworks/HW01/dataset.csv',names=col_names).replace({'label':{'M':1, 'R':0}})
#data = data.sample(frac = 1).reset_index(drop=True)
#data.to_csv('F://Uni/992/Neural Networks/Homeworks/HW01/shuffled.csv', header=False, index = False)


# In[4]:


data = pd.read_csv('F://Uni/992/Neural Networks/Homeworks/HW01/shuffled.csv',names=col_names)


# In[5]:


#data.head()


# In[6]:


x_train = data.loc[:round(0.7*data.shape[0]),:].drop(columns='label')
y_train = data.loc[:round(0.7*data.shape[0]),'label']
x_val = data.loc[round(0.7*data.shape[0]):int(0.8*data.shape[0]),:].drop(columns='label')
y_val = data.loc[round(0.7*data.shape[0]):int(0.8*data.shape[0]),'label']
x_test = data.loc[round(0.8*data.shape[0]):,:].drop(columns='label')
y_test = data.loc[round(0.8*data.shape[0]):,'label']

print('Dataset Shapes:\n',x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# In[7]:


train_corr = x_train.corrwith(y_train, method='pearson')
val_corr = x_val.corrwith(y_val, method='pearson')
test_corr = x_test.corrwith(y_test, method='pearson')


# In[8]:


plt.hist(train_corr, bins=train_corr.size)
plt.title('Train Data Correlation')
plt.ylabel('Number of Features')
plt.xlabel('Correlation Coefficients')
plt.show()


# In[9]:


mostcorr_train = np.where(np.absolute(train_corr)>=0.2)[0]
print('Most Correlated Features with the Training Data:\n', mostcorr_train)
print('Their Correlation Coefficient:\n', train_corr[mostcorr_train])


# In[10]:


plt.hist(val_corr, bins=val_corr.size)
plt.title('Validation Data Correlation')
plt.ylabel('Number of Features')
plt.xlabel('Correlation Coefficients')
plt.show()


# In[11]:


mostcorr_val = np.where(np.absolute(val_corr)>=0.2)[0]
print('Most Correlated Features with the Validation Data:\n', mostcorr_val)
print('Their Correlation Coefficient:\n', val_corr[mostcorr_val])


# In[12]:


plt.hist(test_corr, bins=test_corr.size)
plt.title('Test Data Correlation')
plt.ylabel('Number of Features')
plt.xlabel('Correlation Coefficients')
plt.show()


# In[13]:


mostcorr_test = np.where(np.absolute(test_corr)>=0.2)[0]
print('Most Correlated Features with the Test Data:\n',mostcorr_test)
print('Their Correlation Coefficient:\n', test_corr[mostcorr_test])

