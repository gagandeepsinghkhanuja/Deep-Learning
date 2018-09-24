import numpy as np
import torch
import math


class NeuralNetwork:
    def __init__(self, layers_list: list):
        self.layers_list = layers_list
        if len(self.layers_list) < 2: 
            print("\nERROR: The network must have at least 2 layers!\n")
            exit(1)
        self.numNodes_input = self.layers_list[0]
        self.numNodes_output = self.layers_list[len(self.layers_list) - 1]
        self.theta = {} 
        self.strs = ["" for x in range(len(self.layers_list) - 1)] 
        for i in range(0, len(self.layers_list) - 1):
            self.strs[i] = "theta(layer" + str(i) + "-layer" + str(i + 1) + ")"
        for index in range(len(self.layers_list) - 1):
            self.theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[index]),
                                             (self.layers_list[index] + 1, self.layers_list[index + 1]))
            self.theta[self.strs[index]] = torch.from_numpy(self.theta_np) 


    def getLayer(self, layer: int):
        self.layer = layer
        return self.theta[self.strs[layer]]

    def forward(self, input: torch.DoubleTensor):
        def sigmoid(inp: torch.DoubleTensor):
            product = inp.numpy()  
            sig = 1 / (1 + np.exp(-product))
            return torch.from_numpy(sig) 
        self.input = input
        (row, col) = self.input.size()
        if row != self.numNodes_input:
            print("ERROR: The defined network input layer and input size mismatch!")
            print("Please enter only %r no of inputs" % self.numNodes_input)
            exit(2)
        if col == 1:
            bias = torch.ones((1, 1))
        else:
            bias = torch.ones((1, col))

        bias = bias.type(torch.DoubleTensor)
        sig_prod = self.input

        for i in range(len(self.layers_list) - 1):
            cat_input = torch.cat((bias, sig_prod), 0)
            theta_trans = torch.t(self.theta[self.strs[i]])
            prod = torch.mm(theta_trans, cat_input)
            sig_prod = sigmoid(prod)

        return sig_prod
