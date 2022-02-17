
from meter import Meter
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
from meter import Meter
import torch.nn as nn
import numpy as np
import argparse
import pandas
import torch
import time
import copy
import pdb
import sys
import os



def train_model(model, criterion, optimizer,  scheduler, dset_loaders_new, dset_sizes, logger, saving_path,                                                                                                                                                                                             num_epochs=25, use_gpu=True):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):

        epoch_val = Meter()


        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        #tensordBorad information
        info_tnsrdbrd = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                #optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            counter_data = 0
            for batch_idx, data in enumerate(dset_loaders_new[phase],0):
                # get the inputs
                counter_data = counter_data + 1
                # print('batch_idx = ',batch_idx)

                samples = data['sample']
                categories = data['category']
                #print('categories = ', categories)
                # sys.exit()
                inputs, labels = Variable(samples), Variable(categories)
                labels = labels.type(torch.LongTensor)


                # wrap them in Variable
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.type(torch.cuda.LongTensor)
                else:
                    inputs, labels = inputs, labels

                #print(inputs.shape)
                #sys.exit()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'test':
                    epoch_val.add(preds, labels)
                
                loss = criterion(outputs, labels)
                #if(preds)
                #print('out', preds)
                #print('lab', labels)
                #sys.exit()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase] *100
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                
            
            if phase == 'train':
               #============ TensorBoard logging ============#
               # (1) Log the scalar values
                info = {'train loss': epoch_loss, 'train accuracy': epoch_acc}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)
    
            if phase == 'test' :

                #change learning rate adopt to loss of epoch
                scheduler.step(epoch_loss, epoch) 

                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {'test loss': epoch_loss, 'test accuracy': epoch_acc}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)
                
                # deep copy the model
                if epoch_acc > best_acc:

                    #save model
                    print('Saving Best model...')
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    state = {
                        'model': best_model,
                        'acc':   epoch_acc,
                        'epoch':epoch,
}
                    #if not os.path.isdir('result'):
                        #os.mkdir('result')
                    path = saving_path+ '/densnet-60c50e.t7' 
                    torch.save(best_model, path)
        
                    #calcute confusion matrix
                    confusion = confusion_matrix(epoch_val.label_array, epoch_val.predc_array)
                    confusion = np.around(confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis], 2)

                    #save confusion figure
                    epoch_val.confusion_figure(title= 'Confusion matrix, without normalization', saving_path = saving_path+"/confuMatrix.png")
                    epoch_val.confusion_figure(normalize = True, title= 'Normalized confusion matrix', saving_path = saving_path+"/confuMatrixNorm.png")

        for param_group in optimizer.param_groups:
            print('lr=', param_group['lr'])
            break 
        print()

    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
