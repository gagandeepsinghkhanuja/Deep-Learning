import torch
from torchvision import datasets, transforms
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import time


class MyImg2Num:
    def __init__(self):

        self.train_batch_size = 60
        self.validation_batch_size = 1000 
        self.learning_rate = 0.1 
        self.epochs = 30

        row = 28 
        col = 28 
        self.size1D = row * col 
        self.labels = 10 

        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.train_batch_size, shuffle=True)

        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.validation_batch_size, shuffle=True)

        self.nn_model = NeuralNetwork()  
        self.nn_model.build([self.size1D, 512, 256, 64, self.labels])

    def train(self):
        def training(epoch):
            def onehot_training():
                labels_onehot = torch.zeros(self.train_batch_size,
                                            self.labels) 
                for i in range(self.train_batch_size):
                    labels_onehot[i][target[i]] = 1 
                return labels_onehot

            training_loss = 0 

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                output = self.nn_model.forward(data.view(self.train_batch_size, self.size1D))

                self.nn_model.backward(onehot_training())
                training_loss += self.nn_model.total_loss

                self.nn_model.updateParams(self.learning_rate)

            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)
            print("\nTrain Epoch {}: Average loss: {:.6f}".format(epoch, average_training_loss))
            return average_training_loss

        def validation(epoch):
            def onehot_validation(target):
                labels_onehot = torch.zeros(self.validation_batch_size,
                                            self.labels) 
                for i in range(self.validation_batch_size):
                    labels_onehot[i][target[i]] = 1
                return labels_onehot

            validation_loss = 0
            total_correct = 0 

            for data, target in self.validation_data_loader:
                output = self.nn_model.forward(data.view(self.validation_batch_size, self.size1D))

                validation_loss += ((onehot_validation(target) - output).pow(2).sum()) * 0.5

                value, index = torch.max(output, 1) 
                for i in range(0, self.validation_batch_size):
                    if index[i][0] == target[i]:
                        total_correct += 1

            average_validation_loss = validation_loss / len(self.validation_data_loader.dataset)

            accuracy = 100.0 * total_correct / (len(self.validation_data_loader.dataset))

            print('\nValidation Epoch {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)\n'.
                  format(epoch, average_validation_loss, total_correct, len(self.validation_data_loader.dataset),
                         accuracy))
            return average_validation_loss

        print("\nStarting training of neural network using MyImg2Num on MNIST dataset\n")
        epoch_num = range(1, self.epochs + 1)
        train_loss = list()
        validation_loss = list()
        computation_time = list()

        for i in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss.append(training(i))
            end_time = time.time() - start_time
            computation_time.append(end_time)
            print('\nTrain Epoch {}: Computation Time: {:.2f} seconds'.format(i, end_time))

            validation_loss.append(validation(i))

        plt.figure(1)
        plt.plot(epoch_num, train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(epoch_num, validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        title = 'Loss vs Epochs using NeuralNetwork API (learning rate %r,train batch size %r,validation batch size %r)'\
                % (self.learning_rate, self.train_batch_size, self.validation_batch_size)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

        plt.figure(2)
        plt.plot(epoch_num, computation_time, color='red', linestyle='solid', linewidth='2.0',
                 marker='o', markerfacecolor='red', markersize='5', label='Training Time per epoch')
        plt.ylabel('Computation Time (in seconds)', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        title = 'Computation Time vs Epochs using NeuralNetwork API (learning rate %r,train batch size %r,validation ' \
                'batch size %r)' % (self.learning_rate, self.train_batch_size, self.validation_batch_size)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()


    def forward(self, img: torch.ByteTensor):

        output = self.nn_model.forward(img.view(1, self.size1D)) 
        value, pred_label = torch.max(output, 1)
        return pred_label
