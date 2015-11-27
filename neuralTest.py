# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:39:57 2015

@author: Cristel
"""

import numpy as np

x = np.array(([3,5],[5,1],[10,2]), dtype = float)
y = np.array(([75],[82],[93]), dtype = float)
ynorm = y/np.max(y) 


class NN(object): 
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1 
       
        #weights parameters
        self.W1 = np.random.randn(self.inputLayerSize,\
                        self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,\
                        self.outputLayerSize)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def forward (self, x):
        self.z2 = np.dot(x,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat 
    
    def costFunction(self,x,y):
        self.yHat = self.forward(x)
        J = 0.5 * sum((y - self.yHat)**2)
        return J
    
    def sigmoidPrime(self,z):
        return np.exp(-z)/(1 + np.exp(-z)**2)
    
    def costFunctionPrime(self, x, y):
        self.yHat = self.forward(x)
        
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(x.T, delta2)
        
        return dJdW1, dJdW2
        
        


        