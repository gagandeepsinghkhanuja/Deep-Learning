import numpy as np
import torch
import math


class NeuralNetwork:


    def build(self, layers_list: list):

        self.layers_list = layers_list
        if len(self.layers_list) < 2:
            print("\nERROR: The network must have at least 2 layers!\n")
            exit(1)
        self.numNodes_input = self.layers_list[0]
        self.numNodes_output = self.layers_list[len(self.layers_list) - 1]
        self.L = len(self.layers_list) - 1

        self.Theta = {}
        self.dE_dTheta = {} 
        self.a = {}
        self.z = {}

        self.strs = ["" for x in range(len(self.layers_list) - 1)]
        for i in range(0, len(self.layers_list) - 1):
            self.strs[i] = "Theta(layer" + str(i) + "-layer" + str(i + 1) + ")"

        for index in range(len(self.layers_list) - 1):
            self.Theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[index]),
                                             (self.layers_list[index+1], self.layers_list[index] + 1))
            self.Theta[self.strs[index]] = torch.from_numpy(self.Theta_np).type(torch.FloatTensor)

        self.total_loss = 0.1 

    def getLayer(self, layer: int):
        self.layer = layer
        return self.Theta[self.strs[layer]] 

    def forward(self, input: torch.FloatTensor):
        self.input = input.t()
        (row, col) = self.input.size()

        if row != self.numNodes_input:
            print("ERROR: The defined network input layer and input size mismatch!")
            print("Please enter only %r no of inputs" % self.numNodes_input)
            exit(2)

        if col == 1:
            bias = torch.ones((1, 1))
        else:
            bias = torch.ones((1, col))

        bias = bias.type(torch.FloatTensor)
        self.a[0] = self.input

        for i in range(len(self.layers_list) - 1):
            self.a[i] = torch.cat((bias, self.a[i]), 0)
            theta = self.Theta[self.strs[i]]
            prod = torch.mm(theta, self.a[i]) 
            self.z[i + 1] = prod
            self.a[i + 1] = torch.sigmoid(prod) 

        return self.a[self.L].t()

    def backward(self, target: torch.FloatTensor, loss: str):

        self.target = target.t()
        self.loss = loss
        if loss == 'MSE':
            self.total_loss = ((self.a[self.L] - self.target).pow(2).sum())/(2*(len(target)))

            diff_a = self.a[self.L] * (1 - self.a[self.L]) 
            delta = torch.mul((self.a[self.L] - self.target), diff_a)

            for i in range(self.L-1, -1, -1):
                if i == self.L - 1:
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())
                else:
                    index = torch.LongTensor([1, 2])
                    delta = torch.index_select(delta, 0, index)
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())
                diff_a = self.a[i] * (1-self.a[i])
                x = self.Theta[self.strs[i]].t().mm(delta)
                delta = torch.mul(x, diff_a)



        elif self.loss == 'CE':
            (row, col) = self.target.t().size()
            self.target = self.target.t()

            x = self.a[self.L].t()
            ex = np.exp(x)  
            esum = ex.sum(1)
            b = np.log(esum)
            newsum = (self.a[self.L].t()*self.target).sum(1) 
            sample_loss = b - newsum
            self.total_loss = sample_loss.sum()/row

            delta = self.a[self.L] - self.target

            for i in range(self.L-1, -1, -1):
                if i == self.L - 1:
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())
                else:
                    diff_a = self.a[i+1] * (1-self.a[i+1])
                    delta = self.Theta[self.strs[i]].mm(diff_a)
                    self.dE_dTheta[i] = torch.mm((self.a[self.L] - self.target).sum()/col, delta.t())


    def updateParams(self, eta: float):
        self.eta = eta
        for index in range(len(self.layers_list) - 1):
            gradient = torch.mul(self.dE_dTheta[index], self.eta)
            self.Theta[self.strs[index]] = self.Theta[self.strs[index]] - gradient.t()









