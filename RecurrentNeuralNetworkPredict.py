# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:48:37 2020

@author: Santosh Sah
"""

import pandas as pd
import numpy as np
from RecurrentNeuralNetworkUtils import (readRecurrentNeuralNetworkModel, readRecurrentNeuralNetworkStandardScaler, importRecurrentNeuralNetworkDatasetForTesting,
                                         importRecurrentNeuralNetworkDatasetForTraining, getRecurrentNeuralNetworkTestingSet, getXTest,
                                         saveRecurrentNeuralNetworkRealStockPrice, saveRecurrentNeuralNetworkPredictedStockPrice)

def predict():
    
    recurrentNeuralNetworkDatasetForTesting = importRecurrentNeuralNetworkDatasetForTesting("Recurrent_Neural_Network_Google_Stock_Price_Test.csv")
    recurrentNeuralNetworkDatasetForTraining = importRecurrentNeuralNetworkDatasetForTraining("Recurrent_Neural_Network_Google_Stock_Price_Train.csv")
    
    recurrentNeuralNetworkModel = readRecurrentNeuralNetworkModel()
    recurrentNeuralNetworkStandardScaler = readRecurrentNeuralNetworkStandardScaler()
    
    real_stock_price = getRecurrentNeuralNetworkTestingSet(recurrentNeuralNetworkDatasetForTesting)
    
    #getting the predicted stock price for 2017
    recurrentNeuralNetwork_dataset_total = pd.concat((recurrentNeuralNetworkDatasetForTraining["Open"], recurrentNeuralNetworkDatasetForTesting["Open"]), axis=0)
    
    recurrentNeuralNetwork_input = recurrentNeuralNetwork_dataset_total[len(recurrentNeuralNetwork_dataset_total) - len(recurrentNeuralNetworkDatasetForTesting) - 60:].values
    
    recurrentNeuralNetwork_input = recurrentNeuralNetwork_input.reshape(-1, 1)
    recurrentNeuralNetwork_input = recurrentNeuralNetworkStandardScaler.transform(recurrentNeuralNetwork_input)
    
    X_test = getXTest(recurrentNeuralNetwork_input)
    
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    recurrentNeuralNetwork_predicted_stock_price = recurrentNeuralNetworkModel.predict(X_test)
    
    recurrentNeuralNetwork_predicted_stock_price = recurrentNeuralNetworkStandardScaler.inverse_transform(recurrentNeuralNetwork_predicted_stock_price)
    
    saveRecurrentNeuralNetworkRealStockPrice(real_stock_price)
    saveRecurrentNeuralNetworkPredictedStockPrice(recurrentNeuralNetwork_predicted_stock_price)
    

if __name__ == "__main__":
    predict()