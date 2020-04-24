# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:48:15 2020

@author: Santosh Sah
"""
from sklearn.preprocessing import MinMaxScaler
from RecurrentNeuralNetworkUtils import (importRecurrentNeuralNetworkDatasetForTraining,getRecurrentNeuralNetworkTrainingSet, getXTrainAndYTrain,
                                         saveTrainingDataset, saveRecurrentNeuralNetworkStandardScaler)

def preprocess():
    
    recurrentNeuralNetworkMinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    
    recurrentNeuralNetworkDatasetForTraining = importRecurrentNeuralNetworkDatasetForTraining("Recurrent_Neural_Network_Google_Stock_Price_Train.csv")
    recurrentNeuralNetworkTrainingSet = getRecurrentNeuralNetworkTrainingSet(recurrentNeuralNetworkDatasetForTraining)
    
    recurrentNeuralNetworkMinMaxScaler.fit(recurrentNeuralNetworkTrainingSet)
    saveRecurrentNeuralNetworkStandardScaler(recurrentNeuralNetworkMinMaxScaler)
    
    recurrentNeuralNetworkTrainingSet = recurrentNeuralNetworkMinMaxScaler.transform(recurrentNeuralNetworkTrainingSet)
    
    X_Train, y_Train = getXTrainAndYTrain(recurrentNeuralNetworkTrainingSet)
    
    saveTrainingDataset(X_Train, y_Train)

if __name__ == "__main__":
    preprocess()