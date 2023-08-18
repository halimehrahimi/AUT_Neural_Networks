
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# # Functions

# In[2]:


def predict(s, w, b):
    pred = np.dot(s, w) + b
    pred = [1 if pred>0 else 0]
    return pred


# In[3]:


def perceptron_training(xtrain, ytrain, weights, bias, learn_rate):
    train_pred = []
    for i in range(xtrain.shape[0]):
        s_pred = predict(xtrain.iloc[i], weights, bias)
        error = ytrain.iloc[i]-s_pred
        weights += learn_rate*error*xtrain.iloc[i]
        bias += learn_rate*error
    train_pred = np.dot(xtrain, weights)+bias
    mse = (1/xtrain.shape[0])*(sum((ytrain-train_pred)**2))
    train_pred[train_pred>0] = 1
    train_pred[train_pred<=0] = 0
    tr_error = 1-accuracy_score(ytrain, train_pred)
    tr_mat = confusion_matrix(ytrain, train_pred)
    return weights, bias, mse, tr_error, tr_mat


# In[4]:


def perceptron_train_val(xtrain, ytrain, xval, yval, learn_rate):
    mse_error = []
    train_error = []
    val_error = []
    weights = np.zeros(xtrain.shape[1])
    bias = 0
    t= True
    num_iter = 0
    while t:
        num_iter += 1
        old_weights = np.copy(weights)
        old_bias = np.copy(bias)
        weights, bias, mse, tr_error, train_mat = perceptron_training(xtrain, ytrain, weights, bias, learn_rate)
        mse_error.append(mse)
        train_error.append(tr_error)
        val_pred = []
        for j in range(xval.shape[0]):
            val_pred.append(predict(xval.iloc[j], weights, bias))
        v_error = 1-accuracy_score(yval, val_pred)
        val_error.append(v_error)
        if num_iter==4000 or (all(weights==old_weights) and bias==old_bias):
            t = False
            val_mat = confusion_matrix(yval, val_pred)
    return weights, bias, mse_error, train_error, val_error, num_iter, train_mat, val_mat


# In[5]:


def adaline_training(xtrain, ytrain, weights, bias, learn_rate):
    train_pred = np.zeros(xtrain.shape[0])
    for i in range(xtrain.shape[0]):
        s_pred = np.dot(xtrain.iloc[i], weights) + bias
        error = ytrain.iloc[i]-s_pred
        weights += learn_rate*error*xtrain.iloc[i]
        bias += learn_rate*error
    train_pred = np.dot(xtrain, weights)+bias
    mse = (1/xtrain.shape[0])*(sum((ytrain-train_pred)**2))
    train_pred[train_pred>0] = 1
    train_pred[train_pred<=0] = 0
    tr_error = 1-accuracy_score(ytrain, train_pred)
    tr_mat = confusion_matrix(ytrain, train_pred)
    return weights, bias, mse, tr_error, tr_mat


# In[6]:


def adaline_train_val(xtrain, ytrain, xval, yval, learn_rate):
    train_error = []
    mse_error = []
    val_error = []
    #weights = np.random.random_sample(size=xtrain.shape[1])
    weights = np.zeros(xtrain.shape[1])
    bias = 0
    t= True
    num_iter = 0
    while t:
        num_iter += 1
        old_weights = np.copy(weights)
        old_bias = np.copy(bias)
        weights, bias, mse, tr_error, train_mat = adaline_training(xtrain, ytrain, weights, bias, learn_rate)
        mse_error.append(mse)
        train_error.append(tr_error)
        val_pred = []
        for j in range(xval.shape[0]):
            val_pred.append(predict(xval.iloc[j], weights, bias))
        v_error = 1-accuracy_score(yval, val_pred)
        val_error.append(v_error)
        if num_iter==4000 or (np.max(abs(weights-old_weights))<=0.001):
            t = False
            val_mat = confusion_matrix(yval, val_pred)
    return weights, bias, mse_error, train_error, val_error, num_iter, train_mat, val_mat


# # Preparation

# In[7]:


col_names = []
for c in range(60):
    col_names.append('x%d'%c)
col_names.append('label')


# In[8]:


data = pd.read_csv('F://Uni/992/Neural Networks/Homeworks/HW01/shuffled.csv',names=col_names)


# In[9]:


x_train = data.loc[:round(0.7*data.shape[0]),:].drop(columns='label')
y_train = data.loc[:round(0.7*data.shape[0]),'label']
x_val = data.loc[round(0.7*data.shape[0]):int(0.8*data.shape[0]),:].drop(columns='label')
y_val = data.loc[round(0.7*data.shape[0]):int(0.8*data.shape[0]),'label']
x_test = data.loc[round(0.8*data.shape[0]):,:].drop(columns='label')
y_test = data.loc[round(0.8*data.shape[0]):,'label']

print('Dataset Shapes:\n',x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# In[10]:


x_train2 = pd.concat([x_train, x_train**2], axis=1)
x_val2 = pd.concat([x_val, x_val**2], axis=1)
x_test2 = pd.concat([x_test, x_test**2], axis=1)


# # Perceptron

# In[11]:


learning_rate = 1
weights, bias, mse_error, train_error, val_error, num_iter, train_mat, val_mat = perceptron_train_val(x_train2, y_train, x_val2, y_val, learning_rate)


# In[12]:


plt.plot(np.arange(1,num_iter+1), mse_error)
plt.title('Perceptron Degree 2 - Training')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

plt.plot(np.arange(1,num_iter+1), train_error)
plt.title('Perceptron Degree 2 - Training')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

plt.plot(np.arange(1,num_iter+1), val_error)
plt.title('Perceptron Degree 2 - Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()


# In[13]:


print('Number of Iterations: ', num_iter)
print('Perceptron MSE Error: ', mse_error[-1])
print('Perceptron Train Error:\n', train_error[-1])
print('Perceptron Train Confusion Matrix:\n',train_mat)
print('Perceptron Validation Error:\n', val_error[-1])
print('Perceptron Validation Confusion Matrix:\n', val_mat)
#Test
test_pred = []
for sample in range(x_test2.shape[0]):
    test_pred.append(predict(x_test2.iloc[sample], weights, bias))

test_error = 1-accuracy_score(y_test, test_pred)
test_mat = confusion_matrix(y_test, test_pred)
print('Test Error:\n', test_error)
print('Test Confusion Matrix:\n', test_mat)


# # Adaline

# In[21]:


learning_rate = 0.01
weights, bias, mse_error, train_error, val_error, num_iter, train_mat, val_mat = adaline_train_val(x_train2, y_train, x_val2, y_val, learning_rate)


# In[22]:


plt.plot(np.arange(1,num_iter+1), mse_error)
plt.title('Adeline Degree 2 - Training')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

plt.plot(np.arange(1,num_iter+1), train_error)
plt.title('Adeline Degree 2 - Training')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

plt.plot(np.arange(1,num_iter+1), val_error)
plt.title('Adaline Degree 2 - Validation')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()


# In[23]:


print('Number of Iteration: ', num_iter)
print('Adaline MSE Error:\n', mse_error[-1])
print('Adaline Train Error:\n', train_error[-1])
print('Adaline Train Confusion Matrix:\n',train_mat)
print('Adaline Validation Error:\n', val_error[-1])
print('Adaline Validation Confusion Matrix:\n', val_mat)
#Testing
test_pred = []
for sample in range(x_test2.shape[0]):
    test_pred.append(predict(x_test2.iloc[sample], weights, bias))

test_error = 1-accuracy_score(y_test, test_pred)
test_mat = confusion_matrix(y_test, test_pred)
print('Test Error:\n', test_error)
print('Test Confusion Matrix:\n', test_mat)

