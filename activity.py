from torchvision import datasets, models, transforms
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.optim as optim
from logger import Logger
import torch.nn as nn
import numpy as np
import scipy.io
import torch
import time
import copy
import pdb
import sys
import os



#reead matlab file
#mat = scipy.io.loadmat('sample.mat')

# Set the logger
logger = Logger('./logs')

dataset = {}
first = True
def fillDataset(key, value, dataset):
    if key in dataset:
        #print ('key', key, 'data', dataset)
        dataset[key].append(np.array(value))
    else:
        dataset[key] = [np.array(value)]
    return dataset

#path  = "NTUData"
path  = "test"
files = os.listdir(path)
counter = 0
skipped_count = 0
for file in files:
   counter = counter + 1
   print('Number of files', counter)
   #print('Name of file' + str(file))


   mat = scipy.io.loadmat(path+'//'+file)
   # print mat.shape
   # print mat
   data = mat['samples']
   row, smpls = data.shape
   sampels = {}
   print('shape', data.shape)
   # dataset = {int(i) for i in range(1, 61)}

   # print data
   print('============================================')
   for smpl in range(smpls):
       #print('smpl', smpl)
       val = data[0, smpl]
       if( type(val[0, 0]).__name__ == 'int16'):
           skipped_count = skipped_count + 1
           continue

       #print(type(val[0, 0]))
       #print(len(val[0, 0]))
       #print('============================================')
       val_skl = val['skeleton'][0, 0]
       more_info = {
           'act_clss': val['actionClass'][0, 0][0, 0],
           'setup_number': val['setupNumber'][0, 0][0, 0],
           'camera_id': val['cameraID'][0, 0][0, 0],
           'replication_number': val['replicationNumber'][0, 0][0, 0]
       }

       #print('class', more_info['act_clss'])
       # sys.exit()
       pos, joint, frame = val_skl.shape
       #final_format = np.array([[[int(i) for i in range(3)] for j in range(frame)] for z in range(joint)])
       final_format = np.array([[[int(i) for i in range(224)] for j in range(224)] for z in range(3)])

       # read each frame
       #print('skeleton', val_skl.shape)
       # check size of frame beacuse we have to 224 exactly !!??Question!!??
       if (frame > 224): frame = 224
         
       for fr in range(frame):
           # print ('fr', fr)
           for jnt in range(joint):
               # print ('jnt', jnt)
               # save x,y,z position of joint as RGB format
               # save x as R
               final_format[0, jnt, fr] = np.array(val_skl[0, jnt, fr], dtype='f')
                #save y as G
               final_format[1, jnt, fr] = np.array(val_skl[1, jnt, fr], dtype='f')
               # save z as B
               final_format[2, jnt, fr] = np.array(val_skl[2, jnt, fr], dtype='f')

       
       #final_format = val_skl

       np_skl_final = np.array(final_format)
       # print (final_format)
       saving_path = 'test1.jpg'
       # plt.savefig(saving_path)
       # plt.close(saving_path)

       # save smapel
       sampels[smpl] = final_format
       #print('d', len(dataset))
       #print('a', more_info['act_clss'])
       #create numpy data array
       #first
       if(first):
           data_st = np.array([final_format, more_info['act_clss']])
           #print('f', data_st.size)
       else:
           #print(data_st.size)
           data_st = np.append(data_st, [final_format, more_info['act_clss']])
       dataset = fillDataset(more_info['act_clss'], final_format, dataset)
       first = False

print('============================================')
print('============================================')

sum = 0
#make test and train data
test = []
test_target = []
train = []
train_target = []
for key, val in dataset.items():
    class_count = len(val)
    print( 'number ' + str(class_count))
    print( 'key ' + str(key))
    sum = sum + len(val)
    test_count = class_count//3
    tst_trn = 0
    for item in val:
        tst_trn = tst_trn + 1
        if(tst_trn<test_count):
            test.append(item)
            test_target.append(key)
        else:
            train.append(item)
            train_target.append(key)
dset_sizes = {'train': len(train), 'test': len(test)}
print('len Test', len(test))
print('len Train', len(train))
print(' Number of skipped item ' + str(skipped_count))
print(' sum ' + str(sum))
print(' time ' + str(time.strftime('%X %x %Z')))


#make Dataset

train_new = data_utils.TensorDataset( torch.from_numpy(np.array(train)).float(), torch.from_numpy(np.array(train_target)))
train_loader_new = data_utils.DataLoader(train_new, batch_size=50, shuffle=True)

test_new = data_utils.TensorDataset(torch.from_numpy(np.array(test)).float(), torch.from_numpy(np.array(test_target)))
test_loader_new = data_utils.DataLoader(test_new, batch_size=50, shuffle=True)


print('===================End Load Dataset=========================')
print('=============================================================')

dset_loaders_new = {'train': train_loader_new, 'test': test_loader_new}


# use_gpu = torch.cuda.is_available()
use_gpu = True

print("Using GPU {}".format(use_gpu))

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.

            counter_data = 0
            for batch_idx, (data, target) in enumerate(dset_loaders_new[phase]):
                # get the inputs
                #inputs = data
                #labels = targets_loaders[phase][counter_data]
                counter_data = counter_data + 1
                
                inputs, labels = Variable(data), Variable(target)
                labels = labels.type(torch.LongTensor)


                # wrap them in Variable
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    labels = labels.type(torch.cuda.LongTensor)
                else:
                    inputs, labels = inputs, labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                #print('type', type(inputs))
                #sys.exit()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase] *100
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            #if (epoch+1) % 100 == 0:
        #print ('epoch [%d/%d], Loss: %.4f, Acc: %.2f' %(epoch+1, num_epochs, epoch_loss, epoch_acc))

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            if phase == 'train':
                info = {'loss': epoch_loss, 'accuracy': epoch_acc}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)

       
            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 61)
model_ft.fc.weight.data.normal_(mean=0, std=0.01)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)


###############
#Train Network#
##############
#model_ft.load_state_dict(torch.load('model/res10c20e.pkl'))
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

#############
#Save model#
############
path = 'res18-60c50e100s.pkl'
torch.save(model_ft.state_dict(), path)






