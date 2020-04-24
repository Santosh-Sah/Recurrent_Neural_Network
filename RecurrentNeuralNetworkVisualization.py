# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:49:21 2020

@author: Santosh Sah
"""

import matplotlib.pyplot as plt
from RecurrentNeuralNetworkUtils import (readRecurrentNeuralNetworkRealStockPrice, readRecurrentNeuralNetworkPredictedStockPrice)
"""
Visualizing training set results
"""
def visualisingTrainingSetResult():
    
    recurrentNeuralNetworkRealStockPrice = readRecurrentNeuralNetworkRealStockPrice()
    recurrentNeuralNetworkPredictedStockPrice = readRecurrentNeuralNetworkPredictedStockPrice()
    
    plt.plot(recurrentNeuralNetworkRealStockPrice, color="red", label = "Real Google Stock Price")
    plt.plot(recurrentNeuralNetworkPredictedStockPrice, color="blue", label = "Predicted Google Stock Price")
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock price')
    plt.legend()
    
    plt.savefig("Recurrent_Neural_Networ_Google_Stock_Price_Prediction.png")
    
    plt.show()

if __name__ == "__main__":
    visualisingTrainingSetResult()
