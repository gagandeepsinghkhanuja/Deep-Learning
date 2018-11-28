import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description="Fine Tuning pre-trained AlexNet for classifying Tiny ImageNet dataset")
parser.add_argument('--data', type=str, help='path to directory where tiny imagenet dataset is present')
parser.add_argument('--save', type=str, help='path to directory to save trained model after completion of training')
args = parser.parse_args()


class AlexNet(nn.Module):
    def __init__(self):
        """
        Model Definition
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200)
        )

    def forward(self, inp):
        out = self.features(inp)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out) 
        return out


class TrainModel:
    def __init__(self):
        print("\n\n# ---------- DATA Setup Phase --------- #")
        print("Creating separate folders for each class in validation data and storing images belonging "
              "to each class in corresponding folder")
        print("Completed......................")
        def create_val_folder():
            path = os.path.join(args.data, 'val/images')
            filename = os.path.join(args.data, 'val/val_annotations.txt')
            fp = open(filename, "r") 
            data = fp.readlines()

            val_img_dict = {}
            for line in data:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]
            fp.close()

            for img, folder in val_img_dict.items():
                newpath = (os.path.join(path, folder))
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                if os.path.exists(os.path.join(path, img)): 
                    os.rename(os.path.join(path, img), os.path.join(newpath, img))

        create_val_folder()  

        print("\n\n# ---------- DATALOADER Setup Phase --------- #")
        print("Creating Train and Validation Data Loaders")
        print("Completed......................")

        def class_extractor(class_list):
            filename = os.path.join(args.data, 'words.txt')
            fp = open(filename, "r")
            data = fp.readlines()

            large_class_dict = {}
            for line in data:
                words = line.split("\t")
                super_label = words[1].split(",")
                large_class_dict[words[0]] = super_label[0].rstrip()
            fp.close()

            tiny_class_dict = {} 
            for small_label in class_list:
                for k, v in large_class_dict.items():
                    if small_label == k:
                        tiny_class_dict[k] = v
                        continue

            return tiny_class_dict

        self.train_batch_size = 100 
        self.validation_batch_size = 10 

        train_root = os.path.join(args.data, 'train') 
        validation_root = os.path.join(args.data, 'val/images') 

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_data = datasets.ImageFolder(train_root,
                                          transform=transforms.Compose([transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
        validation_data = datasets.ImageFolder(validation_root,
                                               transform=transforms.Compose([transforms.Scale(256),
                                                                             transforms.CenterCrop(224),
                                                                             transforms.ToTensor(),
                                                                             normalize]))

        self.train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True,
                                                             num_workers=5)

        self.validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                                  batch_size=self.validation_batch_size,
                                                                  shuffle=False, num_workers=5)

        self.class_names = train_data.classes
        self.num_classes = len(self.class_names)
        self.tiny_class = class_extractor(self.class_names) 

        print("\n\n# ---------- MODEL Setup Phase --------- #")
        print("Download pretrained alexnet from torchvision.models, & copy weights to my model except last layer")
        print("Completed......................")

        pretrained_alexnet = models.alexnet(pretrained=True)

        torch.manual_seed(1)
        self.model = AlexNet()

        for i, j in zip(self.model.modules(), pretrained_alexnet.modules()):
            if not list(i.children()):
                if len(i.state_dict()) > 0: 
                    if i.weight.size() == j.weight.size():
                        i.weight.data = j.weight.data
                        i.bias.data = j.bias.data

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

        self.learning_rate = 0.001  
        self.epochs = 50

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.classifier[6].parameters(), lr=self.learning_rate)

        self.epoch_num = range(1, self.epochs + 1)

        filename = 'alexnet_model_checkpoint.pth.tar'
        load_chkpt_file = os.path.join(args.save, filename) 
        if os.path.isfile(load_chkpt_file): 
            print('\nLoading from saved model checkpoint file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            self.start_epoch = chkpt['epoch']
            self.best_accuracy = chkpt['best_accuracy']

            self.model.load_state_dict(chkpt['state_dict'])
            self.optimizer.load_state_dict(chkpt['optimizer'])

            self.train_loss = chkpt['train_loss']
            self.validation_loss = chkpt['validation_loss']
            self.train_accuracy = chkpt['train_accuracy']
            self.validation_accuracy = chkpt['validation_accuracy']
            self.computation_time = chkpt['time']
            print('Completed loading model from checkpoint file: {}, \n(last saved epoch {}, best validation accuracy '
                  '{:.2f})'.format(load_chkpt_file, self.start_epoch, self.best_accuracy))
        else:
            print('\nNo saved model found, no checkpoint to load from\n')
            self.start_epoch = 0
            self.best_accuracy = 0
            self.train_loss = list()
            self.validation_loss = list()
            self.train_accuracy = list()
            self.validation_accuracy = list()
            self.computation_time = list()

    def train(self):


        def save_checkpoint(state, better, file=os.path.join(args.save, 'alexnet_model_checkpoint.pth.tar')):
            torch.save(state, file)
            if better:
                shutil.copyfile(file, os.path.join(args.save, 'alexnet_best_model.pth.tar'))

        def training(epoch: int):

            def onehot_training():

                labels_onehot = torch.zeros(self.train_batch_size, self.num_classes) 
                for i in range(self.train_batch_size): 
                    labels_onehot[i][target[i]] = 1
                return labels_onehot

            self.model.train() 
            training_loss = 0 
            total_correct = 0

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                data, target = Variable(data), Variable(target, requires_grad=False)
                                                                          
                self.optimizer.zero_grad()

                output = self.model(data)

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

            print("\nAverage Training loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".
                  format(average_training_loss, total_correct, len(self.train_data_loader.dataset),
                         self.training_accuracy_cur_epoch))

            return average_training_loss

        def validation(epoch):

            def onehot_validation(target):
                labels_onehot = torch.zeros(self.validation_batch_size, self.num_classes)
                for i in range(self.validation_batch_size): 
                    labels_onehot[i][target[i]] = 1 
                return labels_onehot

            self.model.eval() 
            validation_loss = 0
            total_correct = 0

            for data, target in self.validation_data_loader:
                data, target = Variable(data), Variable(target, requires_grad=False)
                output = self.model(data)
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

        'Actual Code starts here, code above are local methods'

        if self.epochs != self.start_epoch:
            print("\nStarting training of Pre Trained AlexNet network Tiny ImageNet dataset from epoch %r\n" % (
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
                self.best_accuracy = max(self.best_accuracy,
                                         self.validation_accuracy_cur_epoch) 
                print('Saving model checkpoint after completion of epoch {}'.format(i))
                save_checkpoint(
                    {'epoch': i,
                     'best_accuracy': self.best_accuracy,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'train_loss': self.train_loss,
                     'validation_loss': self.validation_loss,
                     'train_accuracy': self.train_accuracy,
                     'validation_accuracy': self.validation_accuracy,
                     'time': self.computation_time,
                     'numeric_class_names': self.class_names,
                     'tiny_class': self.tiny_class,
                     }, better)
                print('Saved, proceeding to next epoch')
                print('------------------------------------------------------------------------------------')
        else:
            print("\nTraining already completed, if you want to train more, increase self.epochs\n")

        print('Average computation time over all iterations {:.2f} seconds\n'.
              format(np.sum(self.computation_time) / self.epochs))

        plt.figure(1)
        plt.plot(self.epoch_num, self.train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(self.epoch_num, self.validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)

        title = 'Loss vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r) ' \
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

        title = 'Computation Time vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam ' \
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

        title = 'Accuracy vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r)' \
                % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

    def forward(self, img: torch.ByteTensor):
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)
        self.model.eval()

        output = self.model(input_image)
        value, pred_label = torch.max(output, 1)
        label = self.tiny_class[self.class_names[pred_label.data[0]]]
        return label

    def view(self, img: torch.ByteTensor, label):
        pred_class = self.forward(img) 
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0)) 

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean

        cv2.namedWindow(pred_class, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pred_class, 640, 480) 
        cv2.imshow(pred_class, img)  
        cv2.waitKey(0) 
        cv2.destroyAllWindows()


if __name__ == '__main__':
    a = TrainModel()
    a.train()
