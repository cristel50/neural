# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:10:53 2015

@author: Cristel
"""

import numpy as np 

x = np.array([[1,0],[1,1]],dtype=float)
y = np.array([[0],[1]], dtype=float) 


class mlpBP(object):
    def __init__(self):    
        self.inputLSize = 2
        self.hiddenLSize = 3
        self.outputLSize = 1
        
        self.W1 = self.inputLSize * self.hiddenLSize
        self.W2 = self.hiddenLSize * self.outputLSize
        
        self.learnRate = 1
        self.moment = 0 
  
#  setting weights        
        self.w11 = 0.2
        self.w12 = 0
        self.w13 = -0.2               
        
        self.w21 = 0
        self.w22 = -0.2
        self.w23 = -0.4

        self.w6y = -0.2
        self.w2y = 0.2
        self.w7y = 0.2
        self.w8y = 0.2
         
        self.W_1 = np.array([[self.w11,self.w12,self.w13],[self.w21,self.w22,self.w23]],dtype = float)        
        
        self.W_d1 = np.array([[1],[0]],dtype = float) 
                
        self.W_2 = np.array([[self.w6y],[self.w7y],[self.w8y]],dtype=float)
        self.W_d2 = np.array([[self.w2y]],dtype=float)


    def sigmoid(self,z):
        return np.array(1/(1+np.exp(-z)))
        
    def forward(self,x): 
        inputNum = np.shape(x)[0]
        for i in range(inputNum):
            self.v1 = np.dot(x[i,:],self.W_1)
            output1s = self.sigmoid(self.v1) 
            output1d = np.dot(x[i,:],self.W_d1)

            #outputs in hidden layer and output
            output1 = np.append(output1s,output1d,axis=0)                        
            output = np.dot(output1,np.append(self.W_2,self.W_d2,axis=0))

            #betas_out
            betaOut = np.zeros((1,self.outputLSize))
            for b in range(self.outputLSize):
                betaOut[b] = (y[i]-output)           
            
            #deltas_from_out  
            delta_w2 = self.learnRate*(np.multiply(betaOut,output1))
            
            #betas_hidden
            print (delta_w2)
     
            self.W_2t = np.append(self.W_2,self.W_d2,axis=0)
            beta_1 = np.multiply(output1,(1-output1)) * betaOut
            print (beta_1)  
#            betaTest = np.dot((np.multiply(output1,(1-output1)) *  betaOut),self.W_2)           
#            for b in range(self.inputLSize):
#                beta_1[0,b] = (output1[b] * (1-output1[b]) * (self.W_2[b]*betaOut))
#
#            #deltas_to_hidden
#            delta_w1 = np.zeros((self.inputLSize,self.hiddenLSize))
#            for r in range(self.inputLSize):
#                for c in range(self.hiddenLSize):
#                    delta_w1[r,c] = beta_1[:,r] * x[i,c]
#            
#            #update weights
#            self.W_1 = np.add(delta_w1.T,self.W_1)
            self.W_2 = np.add(delta_w2.T,np.append(self.W_2,self.W_d2,axis=0))
#            print (self.W_1)
#            print (1)
            print (self.W_2)
                     
#break here to run program with first example        
            break

        
    def costFunction(self,x,y):
        pass
        

neural = mlpBP()
neural.forward(x)



