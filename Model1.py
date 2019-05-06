# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:12:32 2019

@author: Pias Tanmoy
"""



from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
import pandas as pd
from keras.callbacks import ModelCheckpoint


N_EPOCH = 10
BATCH_SIZE = 5
VERBOSE = 1
N_CLASS = 10
OPTIMIZER = Adam()
N_HIDDEN_1 = 128
VALIDATION_SPLIT = 0.1
RESHAPE = 784
DROPOUT = 0.1

Train = pd.read_csv('Train3.csv')
Train = np.array(Train)
Train = shuffle(Train)

X_train = Train[:, :6]
y_train = Train[:, 6:7]

y_train = y_train.astype('int')


INPUT_DIM = X_train.shape[1]
OUTPUT_DIM = np.unique(y_train).shape[0]
N_CLASS = OUTPUT_DIM

print(INPUT_DIM, OUTPUT_DIM)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0],1)


onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

y_train = y_train.astype('int')
y_test = y_test.astype('int')


#X_train, Y_train = shuffle(X_train, y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta

def create_model():
  classifier = Sequential()
  classifier.add(Dense(units = 500, activation='relu', kernel_initializer='glorot_uniform', input_dim=INPUT_DIM))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 100, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 200, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 50, activation='relu', kernel_initializer = 'glorot_uniform'))
  classifier.add(Dropout(DROPOUT))
  classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
  classifier.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
  model = classifier
  
  return model



model = create_model()
BATCH_SIZE = 10
history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, 
                    epochs = 5, verbose = VERBOSE, 
                    validation_split=VALIDATION_SPLIT,
                    shuffle = True)

print(history.history.keys())

N_EPOCH = 30
batch_size_list = []
batch_acc_list = []
batch_val_acc_list = []
batch_train_acc_list = []

selected = [5, 10, 15, 20, 25, 30]



for bs in selected:
  print("Batch size: ", bs)
  model = create_model()
  history = model.fit(X_train, y_train, batch_size = bs, 
            epochs = N_EPOCH, verbose = VERBOSE, 
            validation_split=VALIDATION_SPLIT,
            shuffle = True)
  
  scores = model.evaluate(X_test, y_test, verbose=1)
  
  batch_size_list.append(bs)
  batch_acc_list.append(scores[1])
  batch_train_acc_list.append(history.history['acc'])
  batch_val_acc_list.append(history.history['val_acc'])

np.array(batch_val_acc_list).shape

batch_train_acc_list_np = np.array(batch_train_acc_list)
batch_val_acc_list_np = np.array(batch_val_acc_list)

batch_train_acc_list_np.shape

max_val_acc = []
max_train_acc = []
for i in range(0,6):
  max_val_acc.append(np.amax(batch_val_acc_list_np[i]))
  max_train_acc.append(np.amax(batch_train_acc_list_np[i]))
max_val_acc 

plt.bar(selected, max_val_acc)

from google.colab import files

markers = ['1', '.', '*']
colors = ['#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#e50000', 
          '#95d0fc', '#d1b26f', '#000000', '#029386', '#ff028d',
         '#dbb40c']
for i in range(0, 6):
  plt.plot(batch_train_acc_list_np[i], color = colors[i])
  
plt.legend(batch_size_list, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
           shadow=True, ncol=6, fancybox=True)

plt.title('Training Accuracy (Adagrad)')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.savefig('Train_Accuracy.png', dpi=1000, bbox_inches='tight')
files.download( "Train_Accuracy.png" )

for i in range(0, 6):
  plt.plot(batch_val_acc_list[i], color = colors[i])
  
plt.legend(batch_size_list, loc='upper center', bbox_to_anchor=(0.5, -0.14), shadow=True, ncol=6, fancybox=True)
plt.title('Validation Accuracy (Adagrad)')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')

plt.savefig('Validation_Accuracy.png', dpi=1000, bbox_inches='tight')
files.download( "Validation_Accuracy.png" )

plt.bar?

scores = model.evaluate(X_test, y_test, verbose=1)
print("Test Score: ", scores[0])
print("Accuracy: " , scores[1])

y_pred = model.predict(X_test)
y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

from sklearn.metrics import f1_score
print('sklearn Macro-F1-Score:', f1_score(y_test_argmax, y_pred_argmax, average='macro'))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
y_test_argmax = y_test.argmax(axis=1)
y_pred_argmax = y_pred.argmax(axis=1)

class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test_argmax, y_pred_argmax, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive 
from google.colab import auth 
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()                       
drive = GoogleDrive(gauth)

MODEL_NAME = 'HUMAN_ACTIVITY_2012_98%.h5'
model.save(MODEL_NAME)

classifier_file = drive.CreateFile({'title' : MODEL_NAME})    
classifier_file.SetContentFile(MODEL_NAME)    
classifier_file.Upload()