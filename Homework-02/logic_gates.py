from neural_network import NeuralNetwork
import torch
import numpy as np


class AND:
    def __init__(self):
        self.and_gate = NeuralNetwork([2, 1]) 
        self.theta = self.and_gate.getLayer(0) 
        self.theta.fill_(0)
        self.theta += torch.DoubleTensor([[-30], [20], [20]])
    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()
        bool_value = (output.numpy())
        return bool(np.around(bool_value))
    def forward(self):
        result = self.and_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result


class OR:
    def __init__(self):
        self.or_gate = NeuralNetwork([2, 1])
        self.theta = self.or_gate.getLayer(0)
        self.theta.fill_(0)
        self.theta += torch.DoubleTensor([[-10], [20], [20]])
    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()
        bool_value = (output.numpy())
        return bool(np.around(bool_value))
    def forward(self):
        result = self.or_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result


class NOT:
    def __init__(self):
        self.not_gate = NeuralNetwork([1, 1])
        self.theta = self.not_gate.getLayer(0)
        self.theta.fill_(0) 
        self.theta += torch.DoubleTensor([[10], [-20]])
    def __call__(self, x: bool):
        self.x = x
        output = self.forward()
        bool_value = (output.numpy())
        return bool(np.around(bool_value))
    def forward(self):
        result = self.not_gate.forward(torch.DoubleTensor([[self.x]]))
        return result


class XOR:
    def __init__(self):
        self.xor_gate = NeuralNetwork([2, 2, 1]) 
        self.theta1 = self.xor_gate.getLayer(0)
        self.theta2 = self.xor_gate.getLayer(1)
        self.theta1.fill_(0)
        self.theta1 += torch.DoubleTensor([[-50, -50], [60, -60], [-60, 60]]) 
        self.theta2.fill_(0) 
        self.theta2 += torch.DoubleTensor([[-50], [60], [60]])
    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()
        bool_value = (output.numpy())
        return bool(np.around(bool_value))
    def forward(self):
        result = self.xor_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result
