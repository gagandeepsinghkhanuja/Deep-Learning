import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil, os
import cv2


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.l1 = nn.Linear(16 * 5 * 5, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 100)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out


class img2obj:
    def __init__(self):
        self.train_batch_size = 125
        self.validation_batch_size = 1000
        self.learning_rate = 0.001 
        self.epochs = 50
        channels = 3 
        row = 32 
        col = 32
        self.size1D = channels * row * col
        self.classes = ('apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                        'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle',
                        'caterpillar',
                        'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'kangaroo', 'couch',
                        'crocodile', 'cups', 'crab', 'dinosaur', 'elephant', 'dolphin', 'flatfish', 'forest', 'girl',
                        'fox', 'hamster', 'house', 'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
                        'lizard',
                        'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges',
                        'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 'plates', 'poppies',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm'
                        )
        self.labels = len(self.classes)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
            batch_size=self.train_batch_size, shuffle=True, num_workers=5)

        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=False,
                              transform=transforms.Compose([transforms.ToTensor(), normalize])),
            batch_size=self.validation_batch_size, shuffle=False, num_workers=5)

        torch.manual_seed(1)
        self.nn_model = LeNet5()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        self.epoch_num = range(1, self.epochs + 1)


        load_chkpt_file = 'img2obj_model_checkpoint.pth.tar'
        if os.path.isfile(load_chkpt_file):  
            print('\nLoading from checkpoint file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            self.start_epoch = chkpt['epoch']
            self.best_accuracy = chkpt['best_accuracy']
            self.nn_model.load_state_dict(chkpt['state_dict'])
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.train_loss = chkpt['train_loss']
            self.validation_loss = chkpt['validation_loss']
            self.train_accuracy = chkpt['train_accuracy']
            self.validation_accuracy = chkpt['validation_accuracy']
            self.computation_time = chkpt['time']
            print('Completed loading from checkpoint {}, (last saved epoch {}, best accuracy till now {:.2f})'.
                  format(load_chkpt_file, self.start_epoch, self.best_accuracy))
        else:
            print('\nNo checkpoint to load from\n')
            self.start_epoch = 0
            self.best_accuracy = 0
            self.train_loss = list()
            self.validation_loss = list()
            self.train_accuracy = list()
            self.validation_accuracy = list()
            self.computation_time = list()

    def train(self):
        def save_checkpoint(state, better, file='img2obj_model_checkpoint.pth.tar'):
            torch.save(state, file)
            if better:
                shutil.copyfile(file, 'img2obj_best_model.pth.tar')
        def training(epoch: int):


            def onehot_training():
                labels_onehot = torch.zeros(self.train_batch_size, self.labels)
                for i in range(self.train_batch_size):
                    labels_onehot[i][target[i]] = 1
                return labels_onehot

            self.nn_model.train()
            training_loss = 0
            total_correct = 0
            for batch_id, (data, target) in enumerate(self.train_data_loader):
                data, target = Variable(data), Variable(target, requires_grad=False)                                                                           
                self.optimizer.zero_grad()
                output = self.nn_model(data)

                batch_loss = self.loss_fn(output, target)
                training_loss += batch_loss.data[0]
                batch_loss.backward()
                self.optimizer.step()

                value, index = torch.max(output.data, 1)
                for i in range(0, self.train_batch_size):
                    if index[i] == target.data[i]:
                        total_correct += 1

                print('\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.
                      format(epoch, (batch_id + 1) * self.train_batch_size, len(self.train_data_loader.dataset),
                             100.0 * (batch_id + 1) * self.train_batch_size / len(self.train_data_loader.dataset),
                             batch_loss.data[0]), end="")


            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)


            self.training_accuracy_cur_epoch = 100.0 * total_correct / (len(self.train_data_loader.dataset))
            self.train_accuracy.append(self.training_accuracy_cur_epoch)

            print("\t Average Training loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".
                  format(average_training_loss, total_correct, len(self.train_data_loader.dataset),
                         self.training_accuracy_cur_epoch), end="")

            return average_training_loss

        def validation(epoch):

            def onehot_validation(target):
                labels_onehot = torch.zeros(self.validation_batch_size, self.labels)
                for i in range(self.validation_batch_size): 
                    labels_onehot[i][target[i]] = 1
                return labels_onehot

            self.nn_model.eval()
            validation_loss = 0
            total_correct = 0

            for data, target in self.validation_data_loader:
                data, target = Variable(data), Variable(target, requires_grad=False)
                output = self.nn_model(data)

                batch_loss = self.loss_fn(output, target)
                validation_loss += batch_loss.data[0]

                value, index = torch.max(output.data, 1)
                for i in range(0, self.validation_batch_size):
                    if index[i] == target.data[i]:
                        total_correct += 1

            average_validation_loss = validation_loss / (
                len(self.validation_data_loader.dataset) / self.validation_batch_size)

            self.validation_accuracy_cur_epoch = 100.0 * total_correct / (len(self.validation_data_loader.dataset))
            self.validation_accuracy.append(self.validation_accuracy_cur_epoch)

            print('\nValidation Epoch {}: Average loss: {:.6f} \t Accuracy: {}/{} ({:.2f}%)\n'.
                  format(epoch, average_validation_loss, total_correct, len(self.validation_data_loader.dataset),
                         self.validation_accuracy_cur_epoch))
            return average_validation_loss

        print("\nStarting training of LeNet5 network using img2num on CIFAR100 dataset from epoch %r\n" % (
            self.start_epoch + 1))
        self.training_accuracy_cur_epoch = 0
        self.validation_accuracy_cur_epoch = 0

        for i in range(self.start_epoch + 1, self.epochs + 1):
            start_time = time.time()
            self.train_loss.append(training(i))
            end_time = time.time() - start_time
            self.computation_time.append(end_time)
            print('\t Computation Time: {:.2f} seconds'.format(end_time))

            self.validation_loss.append(validation(i))


            better = self.validation_accuracy_cur_epoch > self.best_accuracy 
            self.best_accuracy = max(self.best_accuracy, self.validation_accuracy_cur_epoch)
            print('Saving checkpoint after completion of epoch {}'.format(i))
            save_checkpoint({'epoch': i,
                             'best_accuracy': self.best_accuracy,
                             'state_dict': self.nn_model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'train_loss': self.train_loss,
                             'validation_loss': self.validation_loss,
                             'train_accuracy': self.train_accuracy,
                             'validation_accuracy': self.validation_accuracy,
                             'time': self.computation_time,
                             }, better)
            print('Saved, proceeding to next epoch')

            print('------------------------------------------------------------------------------------')

        print('Average computation time over all iterations {:.2f} seconds'.
              format(np.sum(self.computation_time) / self.epochs))

        plt.figure(1)
        plt.plot(self.epoch_num, self.train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(self.epoch_num, self.validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)
        title = 'Loss vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r) ' \
                % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()


        plt.figure(2)
        plt.plot(self.epoch_num, self.computation_time, color='red', linestyle='solid', linewidth='2.0',
                 marker='o', markerfacecolor='red', markersize='5', label='Training Time per epoch')
        plt.ylabel('Computation Time (in seconds)', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)

        title = 'Computation Time vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam ' \
                '(learning rate %r) ' % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        plt.figure(3)
        plt.plot(self.epoch_num, self.train_accuracy, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Accuracy')
        plt.plot(self.epoch_num, self.validation_accuracy, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Accuracy')
        plt.ylabel('Accuracy', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)


        title = 'Accuracy vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r)' \
                % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

    def forward(self, img: torch.ByteTensor):

        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)

        self.nn_model.eval()

        output = self.nn_model(input_image)
        value, pred_label = torch.max(output, 1)

        return self.classes[pred_label.data[0]]

    def view(self, img: torch.ByteTensor):
        pred_class = self.forward(img)
        image = img.type(torch.FloatTensor) / 2 + 0.5
        image_numpy = np.transpose(image.numpy(), (1, 2, 0))
        cv2.namedWindow(pred_class, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pred_class, 640, 480) 
        cv2.imshow(pred_class, image_numpy) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def cam(self, idx=0):

        def preprocess(image):
            scaled_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
            preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
            return preprocess(scaled_image)

        cap_obj = cv2.VideoCapture(idx)
        print("\n..............Press q to Quit video capture.............\n")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cap_obj.set(3, 1280)
        cap_obj.set(4, 720)

        while True:
            read, frame = cap_obj.read()

            if read: 
                norm_image_tensor = preprocess(frame)
                pred_class = self.forward(norm_image_tensor)
                cv2.putText(frame, pred_class, (250, 50), font, 2, (255, 200, 100), 5, cv2.LINE_AA)
                cv2.imshow('Webcam Live Video', frame)

            else:
                print('\nError is reading video frame from the webcam..Exiting..')
                break
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        cap_obj.release()
        cv2.destroyAllWindows()
