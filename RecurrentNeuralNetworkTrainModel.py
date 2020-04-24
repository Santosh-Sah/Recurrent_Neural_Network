# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:48:59 2020

@author: Santosh Sah
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
from RecurrentNeuralNetworkUtils import (saveRecurrentNeuralNetworkModel, readRecurrentNeuralNetworkXTrain, readRecurrentNeuralNetworkYTrain)

"""
Train RecurrentNeuralNetwork model 
"""
def trainRecurrentNeuralNetworkModel():
    
    X_train = readRecurrentNeuralNetworkXTrain()
    y_train = readRecurrentNeuralNetworkYTrain()
    
    #reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    #initializing RNN
    recurrentNeuralNetwork = Sequential()
    
    #adding first LSTM layer ans some dropout regularization
    recurrentNeuralNetwork.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    recurrentNeuralNetwork.add(Dropout(0.2))
    
    #adding second LSTM layer and some dropout regularization
    recurrentNeuralNetwork.add(LSTM(units=50, return_sequences=True))
    recurrentNeuralNetwork.add(Dropout(0.2))
    
    #adding third LSTM layer and some dropout regularization
    recurrentNeuralNetwork.add(LSTM(units=50, return_sequences=True))
    recurrentNeuralNetwork.add(Dropout(0.2))
    
    #adding fourth LSTM layer and some dropout regularization
    recurrentNeuralNetwork.add(LSTM(units=50))
    recurrentNeuralNetwork.add(Dropout(0.2))
    
    #adding output layer
    recurrentNeuralNetwork.add(Dense(units=1))
    
    #compile RNN
    recurrentNeuralNetwork.compile(optimizer="adam", loss="mean_squared_error")
    
    #fitting RNN to the training set
    recurrentNeuralNetwork.fit(X_train, y_train, epochs=100, batch_size=32)
     
    #saving the RNN model in a pickle file
    saveRecurrentNeuralNetworkModel(recurrentNeuralNetwork)
    
    

if __name__ == "__main__":
    trainRecurrentNeuralNetworkModel()