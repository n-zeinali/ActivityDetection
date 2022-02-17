from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor
from customDataset import CustomDataset 
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.nn.functional as F
from tabulate import tabulate
import torch.optim as optim
from logger import Logger
import pretrainedmodels
from resnext import *
import torch.nn as nn
import numpy as np
import argparse
import pandas
import torch
import copy
import pdb
import sys
import os
from train import*
from customResnet import*





transform = transforms.Compose([
    ### other PyTorch transforms
    transforms.ToTensor()
    # print('================ToTensor================')
])

parser = argparse.ArgumentParser(description='PyTorch ActivityTraining')

parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--pretrained', default='False', dest='pretrained', action='store_true', help='Use pre-trained model')


parser.add_argument('--data', default='/home/share/NTUFeatures', type=str, help='Path to dataset')
parser.add_argument('--result_dir', default='test', type=str, help='saving directory name')

parser.add_argument('-j', '--workers', default=22, type=int, metavar='N', help='Number of data loading workers (default: 22)')
parser.add_argument('--epochs', '-e', default=45, type=int, metavar='N', help='Number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=50, type=int, metavar='N', help='Mini-batch size (default: 256)')

parser.add_argument('--momentum', '-m', default=0.9, type=float, metavar='M', help='Momentum')
parser.add_argument('--learning_rate', '--lr',  default=0.1, type=float, metavar='learning_rate', help='Initial learning rate')
parser.add_argument('--patience', default=5, type=int, help='Number of epochs with no improvement after which learning rate will be reduced (default: 5)')

parser.add_argument('--use_gpu', default='True', help='Use GPU for Training')

args = parser.parse_args()

print('============================================')
print('============================================')

saving_path = 'result/'+ args.result_dir
os.makedirs('result/'+ args.result_dir)


#o  Set the logger
logger = Logger('./'+ saving_path+ '/logs')


data_dir = args.data

dsets = {x: CustomDataset(data_dir, x, transform=None) for x in ['train', 'test']}
print('Sample image shape: ', (dsets['train'][0]['sample'].shape), end='\n\n')
# print('ctegorys shape: ', type(dsets['train'][0]['category']), end='\n\n')
# sys.exit()

dset_loaders_new = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
                for x in ['train', 'test']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
print('train len = ',dset_sizes['train']) 


# use_gpu = torch.cuda.is_available()
use_gpu = args.use_gpu

print("Using GPU {}".format(use_gpu))



class Densenet121(nn.Module):
    def __init__(self, num_classes = 2, pretrained=False):
        super(Densenet121,self).__init__()
        original_model = models.densenet121(num_classes)
        if(pretrained):
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = (nn.Linear(1024, num_classes))

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        f = F.avg_pool2d(f, kernel_size=7).view(f.size(0), -1)
        y = self.classifier(f)
        return f,y

def getNetwork(args):
    print('============================================')
    print('model=', args.net_type)
    print('============================================')

    if (args.net_type == 'resnet'):
        
        #model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
        #model_ft = pretrainedmodels.__dict__[model_name](num_classes=61)
        #model_ft = inception.inceptionv4() 
        #model.eval()
        #model_ft = models.resnet18(pretrained=False)
        model_ft = resnet34(pretrained=args.pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 61)
        model_ft.fc.weight.data.normal_(mean=0, std=0.01)
    
    elif(args.net_type == 'densnet'):
        #model_ft = models.densenet121(pretrained=args.pretrained)
        dset_classes_number = 61
        model_ft = models.densenet121(num_classes = 61)
        #model_ft.classifier = nn.Linear(1024, dset_classes_number)
        #model_ft.classifier.weight.data.normal_(mean=0, std=0.01)

    elif(args.net_type == 'resnext'):
        model_ft = resnext50(num_classes = 61)
        #model_ft.last_linear = nn.Linear(2048, 61)

    else:
        print('Error : Network should be either [densnet / resnext / resnet]')
        sys.exit(1)
    return model_ft

model_ft = getNetwork(args)
    

if use_gpu:
    model_ft = model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.learning_rate, momentum=args.momentum)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.1)

#monitor epoch loss if epoch loss stop to reduce then change learning rate, divided to 10(lr/10).
scheduler = ReduceLROnPlateau(optimizer_ft, 'min', patience= args.patience)


###############
#Train Network#
###############
#model_ft.load_state_dict(torch.load('model/res10c20e.pkl'))
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, dset_loaders_new, dset_sizes, logger, saving_path, num_epochs=args.epochs, use_gpu=use_gpu)








