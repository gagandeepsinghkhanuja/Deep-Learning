import torch
from torchvision import transforms
from torch.autograd import Variable
import os
import sys
import cv2
import argparse

parser = argparse.ArgumentParser(description="Testing fine-tuned AlexNet for classifying continuous frames from webcam")
parser.add_argument('--model', type=str, help='path to directory where trained model is saved, should be same as the'
                                              ' argument for --save when calling training.py')
args = parser.parse_args()
sys.argv = [sys.argv[0]]

from train import AlexNet


class TestModel:
    def __init__(self):
        self.model = AlexNet()

        filename = 'alexnet_best_model.pth.tar'
        load_chkpt_file = os.path.join(args.model, filename)
        if os.path.isfile(load_chkpt_file):
            print('\nLoading saved model from file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            start_epoch = chkpt['epoch']
            best_accuracy = chkpt['best_accuracy']

            self.model.load_state_dict(chkpt['state_dict'])

            self.class_names = chkpt['numeric_class_names']
            self.tiny_class = chkpt['tiny_class']

            print('Completed loading from file: {}, \n(training stopped after epoch {}, best validation accuracy till '
                  'now {:.2f})'.format(load_chkpt_file, start_epoch, best_accuracy))
        else:
            print('\nNo saved model found, no checkpoint to load from, please train model before testing\n')
            sys.exit(0)

    def forward(self, img: torch.ByteTensor):
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)
        self.model.eval()

        output = self.model(input_image) 
        _, pred_label = torch.max(output, 1)
        label = self.tiny_class[self.class_names[pred_label.data[0]]]
        return label

    def cam(self, idx=0):
        print("\nStarting Webcam to identify live video frames using AlexNet trained on Tiny ImageNet dataset\n")

        def preprocess(image):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize])
            return img_transform(image)

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

                cv2.putText(frame, pred_class, (250, 50), font, 2, (255, 255, 100), 5, cv2.LINE_AA)
                cv2.imshow('Webcam Live Video', frame) 

            else:
                print('\nError is reading video frame from the webcam..Exiting..')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        cap_obj.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    mod = TestModel()
    mod.cam()
