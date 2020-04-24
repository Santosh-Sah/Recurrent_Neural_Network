# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:47:21 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import numpy as np
import pickle

"""
Import training dataset
"""
def importRecurrentNeuralNetworkDatasetForTraining(recurrentNeuralNetworkFileName):
    
    recurrentNeuralNetworkDatasetForTraining = pd.read_csv(recurrentNeuralNetworkFileName)
    
    return recurrentNeuralNetworkDatasetForTraining

"""
Import testing dataset
"""
def importRecurrentNeuralNetworkDatasetForTesting(recurrentNeuralNetworkFileName):
    
    recurrentNeuralNetworkDatasetForTesting = pd.read_csv(recurrentNeuralNetworkFileName)
    
    return recurrentNeuralNetworkDatasetForTesting


def getRecurrentNeuralNetworkTrainingSet(recurrentNeuralNetworkDatasetForTraining):
    
    recurrentNeuralNetworkTrainingSet = recurrentNeuralNetworkDatasetForTraining.iloc[:, 1:2].values
    
    return recurrentNeuralNetworkTrainingSet

def getRecurrentNeuralNetworkTestingSet(recurrentNeuralNetworkDatasetForTesting):
    
    recurrentNeuralNetworkTestingSet = recurrentNeuralNetworkDatasetForTesting.iloc[:, 1:2].values
    
    return recurrentNeuralNetworkTestingSet

def getXTrainAndYTrain(scaledTrainingSet):
    
    X_Train = []
    y_Train = []
    
    for i in range(60, 1258):
        
        X_Train.append(scaledTrainingSet[i - 60:i, 0])
        y_Train.append(scaledTrainingSet[i, 0])
        
    X_Train, y_Train = np.array(X_Train), np.array(y_Train)
        
    return X_Train, y_Train    

def getXTest(scaledTrainingSet):
    
    X_test = []
    
    for i in range(60, 80):
        
        X_test.append(scaledTrainingSet[i - 60:i, 0])
        
    X_test = np.array(X_test)
        
    return X_test    
    
"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveRecurrentNeuralNetworkStandardScaler(recurrentNeuralNetworkStandardScalar):
    
    #Write RecurrentNeuralNetworkStandardScaler in a picke file
    with open("RecurrentNeuralNetworkStandardScaler.pkl",'wb') as RecurrentNeuralNetworkStandardScaler_Pickle:
        pickle.dump(recurrentNeuralNetworkStandardScalar, RecurrentNeuralNetworkStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingDataset(X_train, y_train):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)

"""
Save RecurrentNeuralNetworkModel as a pickle file.
"""
def saveRecurrentNeuralNetworkModel(recurrentNeuralNetworkModel):
    
    #Write RecurrentNeuralNetworkModel as a picke file
    with open("RecurrentNeuralNetworkModel.pkl",'wb') as RecurrentNeuralNetworkModel_Pickle:
        pickle.dump(recurrentNeuralNetworkModel, RecurrentNeuralNetworkModel_Pickle, protocol = 2)

"""
read RecurrentNeuralNetworkStandardScalar from pickel file
"""
def readRecurrentNeuralNetworkStandardScaler():
    
    #load RecurrentNeuralNetworkStandardScaler object
    with open("RecurrentNeuralNetworkStandardScaler.pkl","rb") as RecurrentNeuralNetworkStandardScaler:
        recurrentNeuralNetworkStandardScalar = pickle.load(RecurrentNeuralNetworkStandardScaler)
    
    return recurrentNeuralNetworkStandardScalar

"""
read RecurrentNeuralNetworkModel from pickle file
"""
def readRecurrentNeuralNetworkModel():
    
    #load RecurrentNeuralNetworkModel model
    with open("RecurrentNeuralNetworkModel.pkl","rb") as RecurrentNeuralNetworkModel:
        recurrentNeuralNetworkModel = pickle.load(RecurrentNeuralNetworkModel)
    
    return recurrentNeuralNetworkModel

"""
read X_train from pickle file
"""
def readRecurrentNeuralNetworkXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readRecurrentNeuralNetworkXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readRecurrentNeuralNetworkYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
Save RecurrentNeuralNetworkRealStockPrice as a pickle file.
"""
def saveRecurrentNeuralNetworkRealStockPrice(recurrentNeuralNetworkRealStockPrice):
    
    #Write RecurrentNeuralNetworkRealStockPrice as a picke file
    with open("RecurrentNeuralNetworkRealStockPrice.pkl",'wb') as RecurrentNeuralNetworkRealStockPrice_Pickle:
        pickle.dump(recurrentNeuralNetworkRealStockPrice, RecurrentNeuralNetworkRealStockPrice_Pickle, protocol = 2)

"""
read RecurrentNeuralNetworkRealStockPrice from pickel file
"""
def readRecurrentNeuralNetworkRealStockPrice():
    
    #load RecurrentNeuralNetworkRealStockPrice object
    with open("RecurrentNeuralNetworkRealStockPrice.pkl","rb") as RecurrentNeuralNetworkRealStockPrice:
        recurrentNeuralNetworkRealStockPrice = pickle.load(RecurrentNeuralNetworkRealStockPrice)
    
    return recurrentNeuralNetworkRealStockPrice


"""
Save RecurrentNeuralNetworkPredictedStockPrice as a pickle file.
"""
def saveRecurrentNeuralNetworkPredictedStockPrice(recurrentNeuralNetworkPredictedStockPrice):
    
    #Write RecurrentNeuralNetworkPredictedStockPrice as a picke file
    with open("RecurrentNeuralNetworkPredictedStockPrice.pkl",'wb') as RecurrentNeuralNetworkPredictedStockPrice_Pickle:
        pickle.dump(recurrentNeuralNetworkPredictedStockPrice, RecurrentNeuralNetworkPredictedStockPrice_Pickle, protocol = 2)

"""
read RecurrentNeuralNetworkPredictedStockPrice from pickel file
"""
def readRecurrentNeuralNetworkPredictedStockPrice():
    
    #load RecurrentNeuralNetworkPredictedStockPrice object
    with open("RecurrentNeuralNetworkPredictedStockPrice.pkl","rb") as RecurrentNeuralNetworkPredictedStockPrice:
        recurrentNeuralNetworkPredictedStockPrice = pickle.load(RecurrentNeuralNetworkPredictedStockPrice)
    
    return recurrentNeuralNetworkPredictedStockPrice
