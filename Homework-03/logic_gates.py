from neural_network import NeuralNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt


class AND:
    def __init__(self):
        self.and_gate = NeuralNetwork()
        self.and_gate.build([2, 1])
        self.max_iter = 10000 

    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for AND Gate")
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))
        target_data = torch.unsqueeze(target_data, 1)

        for i in range(self.max_iter):
            index = torch.randperm(4) 
            train_data = torch.index_select(dataset, 0, index)

            for j in range(len(dataset)):
                target_data[j, :] = train_data[j, 0] and train_data[j, 1]

            plt.plot(i, self.and_gate.total_loss, '.r-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            if self.and_gate.total_loss > 0.01: 
                output = self.and_gate.forward(train_data)
                self.and_gate.backward(target_data, 'MSE') 
                self.and_gate.updateParams(1.0)
            else:
                print("Training completed in %d iterations\n" % i)
                break

        old_theta = torch.FloatTensor([[-30, 20, 20]])
        new_theta = self.and_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        output = self.and_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy())
        return bool(np.around(bool_value))


class OR:
    def __init__(self):
        self.or_gate = NeuralNetwork()
        self.or_gate.build([2, 1])
        self.max_iter = 10000


    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for OR Gate")
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))
        target_data = torch.unsqueeze(target_data, 1)

        for i in range(self.max_iter):
            index = torch.randperm(4)
            train_data = torch.index_select(dataset, 0, index)

            for j in range(len(dataset)):
                target_data[j, :] = train_data[j, 0] or train_data[j, 1]

            a = plt.plot(i, self.or_gate.total_loss, '.g-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            if self.or_gate.total_loss > 0.01:
                output = self.or_gate.forward(train_data)
                self.or_gate.backward(target_data, 'MSE')
                self.or_gate.updateParams(5.0)
            else:
                print("Training completed in %d iterations\n" % i)
                break 

        old_theta = torch.FloatTensor([[-10, 20, 20]])
        new_theta = self.or_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        output = self.or_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy())
        return bool(np.around(bool_value))


class NOT:
    def __init__(self):
        self.not_gate = NeuralNetwork()
        self.not_gate.build([1, 1])
        self.max_iter = 10000 


    def __call__(self, x):
        self.x = x
        'Test Case after network is trained'
        output = self.forward(self.x)
        return output


    def train(self):
        print("\nStarting training Network for NOT Gate")
        dataset = torch.FloatTensor([[0], [1]])
        target_data = torch.rand(len(dataset))
        target_data = torch.unsqueeze(target_data, 1)

        for i in range(self.max_iter):

            index = torch.randperm(2)
            train_data = torch.index_select(dataset, 0, index)

            for j in range(len(dataset)):
                target_data[j, :] = not train_data[j, 0]

            plt.plot(i, self.not_gate.total_loss, '.b-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)


            if self.not_gate.total_loss > 0.01: 
                output = self.not_gate.forward(train_data)
                self.not_gate.backward(target_data, 'MSE') 
                self.not_gate.updateParams(5.0)
            else:
                print("Training completed in %d iterations\n" % i) 
                break 


        old_theta = torch.FloatTensor([[10, -20]])
        new_theta = self.not_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool):
        self.x = x
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        output = self.not_gate.forward(torch.FloatTensor([[self.x]]))
        bool_value = (output.numpy())
        return bool(np.around(bool_value))


class XOR:
    def __init__(self):
        self.xor_gate = NeuralNetwork()
        self.xor_gate.build([2, 2, 1])
        self.max_iter = 100000


    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for XOR Gate")
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))
        target_data = torch.unsqueeze(target_data, 1) 

        for i in range(self.max_iter):
            index = torch.randperm(4)
            train_data = torch.index_select(dataset, 0, index)

            for j in range(len(dataset)):
                target_data[j, :] = ((train_data[j, 0]) and (not train_data[j, 1])) or ((not train_data[j, 0]) and (train_data[j, 1]))

            plt.plot(i, self.xor_gate.total_loss, '.m-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            if self.xor_gate.total_loss > 0.01:
                output = self.xor_gate.forward(train_data)
                self.xor_gate.backward(target_data, 'MSE')
                self.xor_gate.updateParams(5.0)
            else:
                print("Training completed in %d iterations\n" % i) 
                break 

        plt.show()
        old_theta1 = torch.FloatTensor([[-50, 60, -60], [-50, -60, 60]])
        old_theta2 = torch.FloatTensor([[-50, 60, 60]])
        new_theta1 = self.xor_gate.getLayer(0)
        new_theta2 = self.xor_gate.getLayer(1)

        print("Manually set Theta: %r %r\n Newly learned Theta %r %r\n" % (old_theta1, old_theta2, new_theta1, new_theta2))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        output = self.xor_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy()) 
        return bool(np.around(bool_value))
