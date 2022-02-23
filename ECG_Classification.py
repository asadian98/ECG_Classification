# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 22:33:54 2021

@author: amirhossein
"""

#%% Load Data: Filtered Signal (.mat) + test_data + labels

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io


Train_mat = scipy.io.loadmat('Matlab_Data.mat') #Load Matlab Filtered Data

data = np.load('train_data.npy')        #Original Data
test = np.load('test_data.npy')         #Test Data
Label = data[:, 10000]                  #Train Data Labels

train_data = data[:, 0:10000].T         #Raw Data
#train_data = Train_mat['TrainData']     #Filtered Data

Fs = 500                                #Sampling Freq.

#%% Feature extraction for Train Data

#import  biosppy (more details: https://biosppy.readthedocs.io/)
from biosppy.signals import ecg
#import NeuroKit (more details: https://neurokit2.readthedocs.io/en/latest/introduction.html)
import neurokit2 as nk

#create train & test features matrix (31 features)
train_features = np.zeros([31, 141])
test_features = np.zeros([31, 20])

#Feature Extraction for each signal
for i in range(141):
    sig = train_data[:, i]
    
    #use biosppy (change show -> True to show the figures) 
    out = ecg.ecg(signal=sig, sampling_rate=Fs, show=False)
    ts = out['ts']
    sig = out['filtered']
    templates_ts = out['templates_ts']
    templates = out['templates']
    template = np.mean(templates, axis = 0)
    heart_rate_ts = out['heart_rate_ts']
    heart_rate = out['heart_rate_ts']
    
    #use NeuroKit / ecg_peaks & ecg_delineate for PRT points
    _, rpeaks = nk.ecg_peaks(sig, sampling_rate=Fs) #just r-peaks
    _, waves_peak = nk.ecg_delineate(sig, rpeaks, sampling_rate=Fs, method="dwt")
    
    L = len(np.array(waves_peak['ECG_P_Offsets']))
    P_Off = np.array(waves_peak['ECG_P_Offsets']).astype(int)[1:L-1]
    P_On = np.array(waves_peak['ECG_P_Onsets']).astype(int)[1:L-1]
    P_Peak = np.array(waves_peak['ECG_P_Peaks']).astype(int)[1:L-1]
    R_Off = np.array(waves_peak['ECG_R_Offsets']).astype(int)[1:L-1]
    R_On = np.array(waves_peak['ECG_R_Onsets']).astype(int)[1:L-1]
    T_Off = np.array(waves_peak['ECG_T_Offsets']).astype(int)[1:L-1]
    T_On = np.array(waves_peak['ECG_T_Onsets']).astype(int)[1:L-1]
    T_Peak = np.array(waves_peak['ECG_T_Peaks']).astype(int)[1:L-1]
    
    T_max = max(sig[T_Peak]) #maximum amplitude of T
    P_max = max(sig[P_Peak]) #maximum amplitude of P
    
    PQRS = np.mean(R_Off - P_On) 
    QRS = np.mean(R_Off - R_On)
    PQRST = np.mean(T_Off - P_On)
    QRST = np.mean(T_Off - R_On)
    TOFFON = np.mean(T_Off - T_On)
    POFFON = np.mean(P_Off - P_On)
    
    rpeaks = rpeaks['ECG_R_Peaks'] #R-points 
    
    signal_var = np.var(sig) #variance 
    signal_mean = np.mean(sig) #mean
    f, Pxx_spec = signal.welch(sig, Fs, scaling='spectrum')
    f_max = f[np.argmax(Pxx_spec)] #maximum freq. in power spectrum
    hist, bin_edges = np.histogram(sig, density=True) #Histogram
    
    R_mean = np.mean(sig[rpeaks]) #mean of maximum amplitude of R
    RR_mean = np.mean(np.diff(rpeaks)) #R to R  
    heart_rate_mean = np.mean(heart_rate) #mean of heart rate
    heart_rate_max = max(heart_rate) #max of heart rate
    peak2peak = max(template) - min(template) #paek to peak 
    
    S = templates_ts[np.argmin(template)] - templates_ts[0] #time - s 
    R = templates_ts[np.argmax(template)] - templates_ts[0] #time - R 
    T = templates_ts[np.argmax(template[np.argmin(template):-1])
                     + np.argmin(template)]- templates_ts[0] #time - T
    Q = templates_ts[np.argmin(template[0:np.argmax(template)])] - templates_ts[0] #time - Q
    P = templates_ts[np.argmax(template[0:np.argmin(template[0:np.argmax(template)])])]
    - templates_ts[0] #time - P
     
    train_features[:, i] = [signal_var, signal_mean, f_max, hist[0], hist[1], hist[2], hist[3], 
               hist[4], hist[5], hist[6], hist[7], hist[8], hist[9], R_mean, RR_mean,
               heart_rate_mean, heart_rate_max, peak2peak, S, R, T, Q, P, QRS, QRST,
               PQRST, T_max, P_max, TOFFON, POFFON, PQRS]

#%% Feature extraction for Test Data

for i in range(20):
    sig = test[i, :]
    
    #use biosppy (change show -> True to show the figures) 
    out = ecg.ecg(signal=sig, sampling_rate=Fs, show=False)
    ts = out['ts']
    sig = out['filtered']
    templates_ts = out['templates_ts']
    templates = out['templates']
    template = np.mean(templates, axis = 0)
    heart_rate_ts = out['heart_rate_ts']
    heart_rate = out['heart_rate_ts']
    
    #use NeuroKit / ecg_peaks & ecg_delineate for PRT points
    _, rpeaks = nk.ecg_peaks(sig, sampling_rate=Fs) #just r-peaks
    _, waves_peak = nk.ecg_delineate(sig, rpeaks, sampling_rate=Fs, method="dwt")
    
    L = len(np.array(waves_peak['ECG_P_Offsets']))
    P_Off = np.array(waves_peak['ECG_P_Offsets']).astype(int)[1:L-1]
    P_On = np.array(waves_peak['ECG_P_Onsets']).astype(int)[1:L-1]
    P_Peak = np.array(waves_peak['ECG_P_Peaks']).astype(int)[1:L-1]
    R_Off = np.array(waves_peak['ECG_R_Offsets']).astype(int)[1:L-1]
    R_On = np.array(waves_peak['ECG_R_Onsets']).astype(int)[1:L-1]
    T_Off = np.array(waves_peak['ECG_T_Offsets']).astype(int)[1:L-1]
    T_On = np.array(waves_peak['ECG_T_Onsets']).astype(int)[1:L-1]
    T_Peak = np.array(waves_peak['ECG_T_Peaks']).astype(int)[1:L-1]
    
    T_max = max(sig[T_Peak]) #maximum amplitude of T
    P_max = max(sig[P_Peak]) #maximum amplitude of P
    
    PQRS = np.mean(R_Off - P_On) 
    QRS = np.mean(R_Off - R_On)
    PQRST = np.mean(T_Off - P_On)
    QRST = np.mean(T_Off - R_On)
    TOFFON = np.mean(T_Off - T_On)
    POFFON = np.mean(P_Off - P_On)
    
    rpeaks = rpeaks['ECG_R_Peaks'] #R-points 
    
    signal_var = np.var(sig) #variance 
    signal_mean = np.mean(sig) #mean
    f, Pxx_spec = signal.welch(sig, Fs, scaling='spectrum')
    f_max = f[np.argmax(Pxx_spec)] #maximum freq. in power spectrum
    hist, bin_edges = np.histogram(sig, density=True) #Histogram
    
    R_mean = np.mean(sig[rpeaks]) #mean of maximum amplitude of R
    RR_mean = np.mean(np.diff(rpeaks)) #R to R  
    heart_rate_mean = np.mean(heart_rate) #mean of heart rate
    heart_rate_max = max(heart_rate) #max of heart rate
    peak2peak = max(template) - min(template) #paek to peak 
    
    S = templates_ts[np.argmin(template)] - templates_ts[0] #time - s 
    R = templates_ts[np.argmax(template)] - templates_ts[0] #time - R 
    T = templates_ts[np.argmax(template[np.argmin(template):-1])
                     + np.argmin(template)]- templates_ts[0] #time - T
    Q = templates_ts[np.argmin(template[0:np.argmax(template)])] - templates_ts[0] #time - Q
    P = templates_ts[np.argmax(template[0:np.argmin(template[0:np.argmax(template)])])]
    - templates_ts[0] #time - P
     
    test_features[:, i] = [signal_var, signal_mean, f_max, hist[0], hist[1], hist[2], hist[3], 
               hist[4], hist[5], hist[6], hist[7], hist[8], hist[9], R_mean, RR_mean,
               heart_rate_mean, heart_rate_max, peak2peak, S, R, T, Q, P, QRS, QRST,
               PQRST, T_max, P_max, TOFFON, POFFON, PQRS]
    
#%% normalize data z = (x - u) / s
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_features.T)
# Apply transform to both the training set and the test set.
x_train_normalized = scaler.transform(train_features.T)
x_test_normalized = scaler.transform(test_features.T)

#%% Visualize 3D Projection

from sklearn.decomposition import PCA
import pandas as pd

#create dataframe
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x_train_normalized)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
df = pd.DataFrame(data = Label, columns = ['target'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

import matplotlib.cm as cm

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = range(41)
colors = cm.rainbow(np.linspace(0, 1, len(targets))) #create colors
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets, prop={'size': 6})
ax.grid()

#%% PCA with all the features / change to PCA(0.95) for 0.95 of variance

pca = PCA()
pca.fit(x_train_normalized)

print("\n\nNumber of components used")
print(pca.n_components_)

#Project Train & Test Data
train_p = pca.transform(x_train_normalized)
test_p = pca.transform(x_test_normalized)

#%% Model Selection and Training

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#SVM parametes for GridSearch
SVC_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree': range(10)}]

clf_SVC = GridSearchCV(SVC(), SVC_parameters)

#KNN parametes for GridSearch
KNN_parameter = [{'n_neighbors': range(41)}]
clf_KNN = GridSearchCV(KNeighborsClassifier(), KNN_parameter)

#Classifiers
classifiers = [
    clf_KNN,
    LogisticRegression(),
    clf_SVC,
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LinearDiscriminantAnalysis()]

#Each time train a classifier and add the score
scores = []
for clf in classifiers:
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores.append(np.mean(cross_val_score(clf, train_p, Label, scoring='accuracy'
                                          , cv=cv, n_jobs=-1)))

#Print the result
print("\n\n********** Result **********\n")
print("The best classifier is ", classifiers[np.argmax(scores)])
print("\naccuracy = ", max(scores))
best_clf = classifiers[np.argmax(scores)].fit(train_p, Label) #Train with all PCA component 
test_target = best_clf.predict(test_p) #Calculate test targets

#%% Save Test targets
from numpy import save
save('TestLabels.npy', test_target)