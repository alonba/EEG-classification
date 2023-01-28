# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:55:08 2022

@author: Tamar Shavit
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
tf.config.run_functions_eagerly(True)
import mne


def product(*args, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def plot_confusion_matrix(cm, class_names,flag):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm_norm.max() / 2.
    
    for i, j in product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        color = "white" if cm_norm[i, j] > 0.34 else "black"
        #color = "white" if cm_norm[i, j] > threshold else "black"
        plt.text(j, i, '({:.0f}%)\n {}'.format(cm_norm[i, j]*100, cm[i,j]), horizontalalignment="center", color=color, linespacing=3, fontsize='large')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

#####  Import my matlab data
# itay_data = mne.io.read_epochs_eeglab(r"C:\Users\alonb\OneDrive - Technion\טכניון\סמסטר ט' - אביב 2022\פרוייקט א\מאטלב\08_31_10_Itay\Training\EEG_SET\62_base_removed_short_filtered_bad_epochs_removed_locs.set")
# data_right = np.moveaxis(itay_data["Right"].get_data(), 0, -1)
# data_left = np.moveaxis(itay_data["Left"].get_data(), 0, -1)
# data_nomove = np.moveaxis(itay_data["NoMove"].get_data(), 0, -1)

#####  Import part of Igor's data (portion equal to one subject's data ~ 450 epochs)
# path = 'data/igor/'
# data_right = sio.loadmat(os.path.join(path,'0.mat'))["EEG"][0,0]["data"][:,:,:150]
# data_left = sio.loadmat(os.path.join(path,'1.mat'))["EEG"][0,0]["data"][:,:,:150]
# data_nomove = sio.loadmat(os.path.join(path,'2.mat'))["EEG"][0,0]["data"][:,:,:150]

####### Get Igor's .set files and extract the data from them
data_right = mne.io.read_epochs_eeglab('data/igor/S1_33_64_RLS_clean.set').get_data() * (10**6)
data_right = np.moveaxis(data_right, 0, -1)
data_left = mne.io.read_epochs_eeglab('data/igor/S2_33_64_RLS_clean.set').get_data() * (10**6)
data_left = np.moveaxis(data_left, 0, -1)
data_nomove = mne.io.read_epochs_eeglab('data/igor/S4_33_64_RLS_clean.set').get_data() * (10**6)
data_nomove = np.moveaxis(data_nomove, 0, -1)

###### LOADING DATA - data from Igor, after cutting into epochs of [-200,800]msec and manually removing outliers
# path = 'data/igor/'
# data_right = sio.loadmat(os.path.join(path,'0.mat'))["EEG"][0,0]["data"]
# data_left = sio.loadmat(os.path.join(path,'1.mat'))["EEG"][0,0]["data"]
# data_nomove = sio.loadmat(os.path.join(path,'2.mat'))["EEG"][0,0]["data"]

# elec_names = ['FP1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8','FP2','AF3','AFz','F1','F5','FT7','FC3','FCz','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4']
elec_names = ['Fp1','Fz','F3','F7','FT9','FC5','FC1','C3','T7','TP9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','p8','TP10','CP6','CP2','Cz','C4','T8','FT10','FC6','FC2','F4','F8',      'AF3','AFz','F1','F5','FT7','FC3','FCz','C1','C5','TP7','CP3','P1','P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','CP4','TP8','C6','C2','FC4','FT8','F6','F2','AF4','AF8']
# SANITY CHECK - testing if the data is valid:
def plot_avg_v(elec_idx, data_left, data_right, data_nomove, elec_name):
    time = np.arange(-200, 800, 2)
    events_type = ['right','left','no movement']
    
    left_avg = np.mean(data_left[elec_idx,:,:], axis=1)
    right_avg = np.mean(data_right[elec_idx,:,:], axis=1)
    nomove_avg = np.mean(data_nomove[elec_idx,:,:], axis=1)
    
    plt.figure(figsize=(7,3))
    plt.plot(time, right_avg)
    plt.plot(time, left_avg)
    plt.plot(time, nomove_avg)
    plt.legend(events_type)
    plt.xlabel('Time (msec)')
    plt.ylabel(r'$\mu$V')
    plt.title(f'{elec_name[elec_idx]} electrode')

    plt.show()
# elec_desired = 'FC2'#'FC2'
# elec_idx = elec_names.index(elec_desired)#11 #electrode number
# plot_avg_v(elec_idx, data_left, data_right, data_nomove, elec_names)

###### ORGANIZE AND SHUFFLE - organizing all data in one array
data_orig = np.concatenate([data_right, data_left, data_nomove], axis=2)
data_org = np.moveaxis(data_orig, 2, 0)
labels_org = np.concatenate([np.zeros((data_right.shape[2])), np.ones((data_left.shape[2])), 2*np.ones((data_nomove.shape[2]))])

data, labels = shuffle(data_org, labels_org, random_state=42)

# RELEVANT ELECTRODES AND TIME - cutting 35 center electrodes and 700 msec
# 5 rows, 7 cols, left -> right and up -> down
elec_order = ['F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5','P3','P1','Pz','P2','P4','P6']
elec_idx = [elec_names.index(x) for x in elec_order]

# time cut of 150: ==> 100msec:800msec
data_relevant = data[:,elec_idx,150:] 

n_trials = np.shape(data_relevant)[0]
n_samples = np.shape(data_relevant)[2]

data_3d = np.reshape(data_relevant, (n_trials,5,7,n_samples))

labels_onehot = np.zeros((n_trials, 3))
for i in range(n_trials):
    labels_onehot[i, int(labels[i])]=1

X_train_val, X_test, y_train_val, y_test = train_test_split(data_3d, labels_onehot, test_size=0.1, random_state=42, stratify=labels_onehot)

# for conv3d add additional axis (used as single channel):
X_train_val = np.expand_dims(X_train_val,axis=4)
X_test = np.expand_dims(X_test,axis=4)

###### Model (Igor's)
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(20, (3,3,200), padding='valid',input_shape=(5, 7, n_samples, 1), activation="relu"),
    tf.keras.layers.Conv3D(10, (2,2,20), padding='valid', activation="relu"),
    tf.keras.layers.MaxPooling3D((2, 2, 50), strides=(1,1,50)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Training
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val)
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))

# Evaluate the validation score:
scores = model.evaluate(X_val, y_val)
print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

###### Predicting
prediction = model.predict(X_test)
y_hat = np.zeros(np.shape(y_test))
for i in range(np.shape(y_hat)[0]):
    y_hat[i,np.argmax(prediction[i,:])] = 1
delta = np.sum(abs(y_hat - y_test), axis=1)
accuracy = len(delta[delta==0])/len(delta)*100

# Plot Accuracy graph (train set and val set)
metrics_df = pd.DataFrame(history.history)
ax = metrics_df[["accuracy","val_accuracy"]].plot(title="Accuracy");
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy")
ax.legend(["Training accuracy","Validation accuracy"])

# Plot Loss graph (train set and val set)
axe = metrics_df[["loss","val_loss"]].plot(title="Loss");
axe.set_xlabel("Epochs")
axe.set_ylabel("Loss Value")
axe.set_title("Loss")
axe.legend(["Training loss","Validation loss"])

# Calculating Confusion Matrix
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)

class_names=["Right","Left","No Movement"]
figure = plot_confusion_matrix(cm, class_names=class_names, flag=1)
x = 'end of file'