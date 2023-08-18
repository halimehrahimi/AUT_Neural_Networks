
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt


# In[2]:


"""data = pd.read_csv('/content/drive/MyDrive/ANN/Datasets/YearPredictionMSD.txt', header=None)
data = data.sample(frac = 1).reset_index(drop=True)
data.to_csv('/content/drive/MyDrive/ANN/Datasets/shuffled.csv', header=False, index = False)"""


# In[3]:


data = pd.read_csv('/content/drive/MyDrive/ANN/Datasets/shuffled.csv', header=None)
data.head()


# In[4]:


data.describe()


# In[5]:


#scaler = StandardScaler()
#data = pd.DataFrame(scaler.fit_transform(data))
datanorm = (data-data.min())/(data.max()-data.min())
xdata = datanorm.drop(columns=0)
#xdata = data.drop(columns=0)
#xdata = pd.DataFrame(scaler.fit_transform(xdata))
#xdata = (xdata-xdata.min())/(xdata.max()-xdata.min())
ydata = datanorm.loc[:,0]
#ydata = data.loc[:,0]
#ydata = ((ydata-ydata.min())*10)/(ydata.max()-ydata.min())
xdata.head()


# In[6]:


x_train = xdata.loc[:round(0.7*data.shape[0]),:]
y_train = ydata.loc[:round(0.7*data.shape[0])]
x_val = xdata.loc[round(0.7*data.shape[0]):int(0.9*data.shape[0]),:]
y_val = ydata.loc[round(0.7*data.shape[0]):int(0.9*data.shape[0])]
x_test = xdata.loc[round(0.9*data.shape[0]):,:]
y_test = ydata.loc[round(0.9*data.shape[0]):]

print('Dataset Shapes:\n',x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# In[7]:


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
  def __init__(self, patience=0):
    super(EarlyStoppingCallback, self).__init__()
    self.patience = patience

  def on_train_begin(self, logs=None):
    self.best = np.Inf
    self.wait = 0
    self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current_loss = logs.get("val_mean_squared_error")
    if np.less(current_loss, self.best):
      self.best = current_loss
      self.wait = 0
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print("epoch: %d: early stopping" % self.stopped_epoch)


# In[8]:


tf.keras.backend.clear_session()

reg_callback = EarlyStoppingCallback(patience=5)

def adapt_learning_rate(epoch):
    if epoch>5:
      return 0.0002 / (epoch-4)
    else:
      return 0.0002
  
my_lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adapt_learning_rate)

inp = tf.keras.Input(shape = 90, name="input")
hidden = tf.keras.layers.Dense(256)(inp)
hidden = tf.keras.layers.Dense(128)(hidden)
#hidden = tf.keras.layers.Dense(512)(hidden)
outp = tf.keras.layers.Dense(1)(hidden)

model = tf.keras.Model(inputs=inp, outputs=outp)


# In[9]:


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['mean_squared_error'])
history = model.fit(x_train, y_train, epochs=1000, verbose=1, validation_data=(x_val,y_val), callbacks=[reg_callback, my_lr_scheduler])


# In[10]:


test_pred = np.round((model.predict(x_test) * (2011-1922)) + 1922)
#test_pred = np.round(model.predict(x_test))
#test_pred = np.round(model.predict(x_test))+1922
#test_pred = np.round(model.predict(x_test)*(9/10))+1922


# In[11]:


ytestnotnorm = data.loc[round(0.9*data.shape[0]):,0].astype(int)
print(ytestnotnorm)
print(test_pred)
confmat = confusion_matrix(ytestnotnorm, test_pred, np.arange(1922,2012))
print(confmat)
df_confmat = pd.DataFrame(confmat, index = [i for i in np.arange(1922,2012)], columns = [i for i in np.arange(1922,2012)])


# In[12]:


plt.figure(figsize = (20,20))
sn.heatmap(df_confmat)


# In[13]:


accuracy_score(ytestnotnorm, test_pred)


# In[14]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[15]:


tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)

