
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


xdata = data.drop(columns=0)
ydata = data.loc[:,0]
scaler = StandardScaler()
xdata = pd.DataFrame(scaler.fit_transform(xdata))
xdata.head()


# In[5]:


x_train = xdata.loc[:round(0.7*data.shape[0]),:]
y_train = ydata.loc[:round(0.7*data.shape[0])]
x_val = xdata.loc[round(0.7*data.shape[0]):int(0.9*data.shape[0]),:]
y_val = ydata.loc[round(0.7*data.shape[0]):int(0.9*data.shape[0])]
x_test = xdata.loc[round(0.9*data.shape[0]):,:]
y_test = ydata.loc[round(0.9*data.shape[0]):]

print('Dataset Shapes:\n',x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# In[6]:


y_train = tf.keras.utils.to_categorical(y_train-1922, 90)
y_val = tf.keras.utils.to_categorical(y_val-1922, 90)
y_test = tf.keras.utils.to_categorical(y_test-1922, 90)


# In[7]:


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
  def __init__(self, patience=0):
    super(EarlyStoppingCallback, self).__init__()
    self.patience = patience

  def on_train_begin(self, logs=None):
    self.best = -np.Inf
    self.wait = 0
    self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current_accuracy = logs.get("val_accuracy")
    if np.greater(current_accuracy, self.best):
      self.best = current_accuracy
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


def adapt_learning_rate(epoch):
    if epoch>5:
      return 0.0005 / (epoch-4)
    else:
      return 0.0005
  
my_lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adapt_learning_rate)


# In[9]:


tf.keras.backend.clear_session()

cls_callback = EarlyStoppingCallback(patience=5)

inp = tf.keras.Input(shape = 90, name="input")
hidden = tf.keras.layers.Dense(128, activation='relu')(inp)
hidden = tf.keras.layers.Dense(512, activation='relu')(hidden)
#hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
outp = tf.keras.layers.Dense(90, activation='softmax')(hidden)

model = tf.keras.Model(inputs=inp, outputs=outp)


# In[10]:


model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=1000, verbose=1, validation_data=(x_val,y_val), callbacks=[cls_callback, my_lr_scheduler])


# In[18]:


test_pred = np.argmax(model.predict(x_test), axis=-1)+1922
confmat = confusion_matrix(ydata.loc[round(0.9*data.shape[0]):], test_pred, np.arange(1922,2012))
print(confmat)
print(confmat.shape)
df_confmat = pd.DataFrame(confmat, index = [i for i in np.arange(1922,2012)], columns = [i for i in np.arange(1922,2012)])


# In[19]:


plt.figure(figsize = (20,20))
sn.heatmap(df_confmat)


# In[21]:


accuracy_score(ydata.loc[round(0.9*data.shape[0]):], test_pred)


# In[23]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[26]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[24]:


tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True)


# In[25]:


model.summary()

