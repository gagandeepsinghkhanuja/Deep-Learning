import numpy as np
import torch
import math


class NeuralNetwork:

    def __init__(self):
        pass

    def build(self, layers_list: list):
        self.layers_list = layers_list
        self.numLayers = len(self.layers_list)
        if self.numLayers < 2:  # Error Checking
            print("\nERROR: The network must have at least 2 layers!\n")
            exit(1)

        self.L = self.numLayers - 1  

        self.Theta = {}
        self.dE_dTheta = {}
        self.a = {}
        self.z = {}

        for i in range(0, self.numLayers - 1): 
            theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[i]),
                                        (self.layers_list[i + 1], self.layers_list[i] + 1))
            self.Theta[i] = torch.from_numpy(theta_np).type(torch.FloatTensor)

        self.total_loss = 0.0  

    def getLayer(self, layer: int):
        return self.Theta[layer] 

    def forward(self, input: torch.FloatTensor):
        self.input = input.t() 
        (row, col) = self.input.size() 

        if row != self.layers_list[0]:
            print("ERROR: The defined network input layer and input size mismatch!")
            print("Please enter only %r no of inputs" % self.layers_list[0])
            exit(2)

        if col == 1:
            bias = torch.ones((1, 1))
        else:
            bias = torch.ones((1, col))  

        bias = bias.type(torch.FloatTensor) 
        self.a[0] = self.input  

        for l in range(0, self.numLayers - 1):  
            self.a[l] = torch.cat((bias, self.a[l]), 0) 
            self.z[l + 1] = torch.mm(self.Theta[l], self.a[l]) 
            self.a[l + 1] = torch.sigmoid(self.z[l + 1])  

        return self.a[self.L].t()

    def backward(self, target: torch.FloatTensor):
        self.target = target.t()
        (row, col) = self.target.size()

        self.total_loss = ((self.a[self.L] - self.target).pow(2).sum()) / (2 * col) 
        diff_a = self.a[self.L] * (1 - self.a[self.L])
        delta = torch.mul((self.a[self.L] - self.target), diff_a)

        for l in range(self.numLayers - 2, -1, -1):
            if l == self.numLayers - 2: 
                self.dE_dTheta[l] = torch.mm(delta, self.a[
                    l].t())

            else: 
                delta = delta.narrow(0, 1, delta.size(0) - 1) 
                self.dE_dTheta[l] = torch.mm(delta, self.a[l].t()) 

            diff_a = self.a[l] * (1 - self.a[l])
            delta = torch.mul(self.Theta[l].t().mm(delta),
                              diff_a) 

    def updateParams(self, eta: float):
        for index in range(0, self.numLayers - 1):
            gradient = torch.mul(self.dE_dTheta[index], eta)
            self.Theta[index] = self.Theta[index] - gradient
