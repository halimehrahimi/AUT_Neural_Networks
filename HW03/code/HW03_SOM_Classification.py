
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


# In[2]:


def create_data_set(): 
    scaler = StandardScaler()
    data = pd.read_csv('/content/drive/MyDrive/ANN/Datasets/yale_train.csv', header=None)
    train_label = data.iloc[:,-1]
    train_data = data.iloc[:,:-1]
    train_data = pd.DataFrame(scaler.fit_transform(train_data))
    data = pd.read_csv('/content/drive/MyDrive/ANN/Datasets/yale_val.csv', header=None)
    val_label = data.iloc[:,-1]
    val_data = data.iloc[:,:-1]
    val_data = pd.DataFrame(scaler.transform(val_data))
    data = pd.read_csv('/content/drive/MyDrive/ANN/Datasets/yale_test.csv', header=None)
    test_label = data.iloc[:,-1]
    test_data = data.iloc[:,:-1]
    test_data = pd.DataFrame(scaler.fit_transform(test_data))

    print('Number data in each group: ', len(train_label), len(val_label), len(test_label))
    
    return np.array(train_data), np.array(train_label), np.array(val_data), np.array(val_label), np.array(test_data), np.array(test_label)


# In[3]:


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
  def __init__(self, patience=0):
    super(EarlyStoppingCallback, self).__init__()
    self.patience = patience

  def on_train_begin(self, logs=None):
    self.best = -np.Inf
    self.wait = 0
    self.stopped_epoch = 0

  def on_epoch_end(self, epoch, logs=None):
    current_accuracy = logs.get("accuracy")
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


# In[4]:


if __name__ == "__main__":
    map_size = [11,11]
    train_data, train_label, val_data, val_label, test_data, test_label = create_data_set()

    train_label = tf.keras.utils.to_categorical(train_label, 11)
    val_label = tf.keras.utils.to_categorical(val_label, 11)

    tf.keras.backend.clear_session()
    cls_callback = EarlyStoppingCallback(patience=5)

    inp = tf.keras.Input(shape = (map_size[0]*map_size[1]), name="input")
    hidden = tf.keras.layers.Dense(128, activation='relu')(inp)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outp = tf.keras.layers.Dense(11, activation='softmax')(hidden)

    model = tf.keras.Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(train_data, train_label, epochs=25, verbose=1, validation_data=(val_data,val_label))#, callbacks=[cls_callback]


# In[5]:


test_pred = np.argmax(model.predict(test_data), axis=-1)
print(confusion_matrix(test_label, test_pred, np.arange(0,11)))
print(accuracy_score(test_label, test_pred))


# In[6]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

