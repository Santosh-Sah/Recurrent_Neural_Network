# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:48:57 2020

@author: Santosh Sah
"""

from sklearn.metrics import mean_squared_error
import math
from RecurrentNeuralNetworkUtils import (readRecurrentNeuralNetworkRealStockPrice, readRecurrentNeuralNetworkPredictedStockPrice)

"""

calculating Recurrent Neural Network Mean Squared Error

"""
def testRecurrentNeuralNetworkMeanSquaredError():
    
    recurrentNeuralNetworkRealStockPrice = readRecurrentNeuralNetworkRealStockPrice()
    recurrentNeuralNetworkPredictedStockPrice = readRecurrentNeuralNetworkPredictedStockPrice()
    
    rmse = math.sqrt(mean_squared_error(recurrentNeuralNetworkRealStockPrice, recurrentNeuralNetworkPredictedStockPrice))
    
    print("rmse is: ", str(rmse)) #11.73
    
if __name__ == "__main__":
    testRecurrentNeuralNetworkMeanSquaredError()
    