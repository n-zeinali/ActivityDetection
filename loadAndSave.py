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
import cv2
import sys
import pickle



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
path  = "./projects/test100"
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
       
   
       
       final_format = np.array(val_skl)
       # print(final_format.shape)
       # h,w = final_format.shape
       # vis2 = cv2.CreateMat(h, w, cv.CV_32FC3)
       h,w,d = final_format.shape
       img = np.zeros([w,d,h])

       img[:,:,0] = final_format[0,:,:]
       img[:,:,1] = final_format[1,:,:]
       img[:,:,2] = final_format[2,:,:]
       res = cv2.resize(img,(224,224),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
       # print(res.shape)
       # sys.exit()
       resImg = np.zeros([h,224,224])
       resImg[0,:,:] = res[:,:,0]       
       resImg[1,:,:] = res[:,:,1]  
       resImg[2,:,:] = res[:,:,2]  
       # print(resImg.shape)
       # cv2.imshow("WindowNameHere", res)
       # cv2.waitKey(0)
       #np_skl_final = np.array(final_format)
       # final_format.resize(3, 224, 224)      
       # print (final_format)
       # saving_path = 'test1.jpg'
       # plt.savefig(saving_path)
       # plt.close(saving_path)

       # save smapel
       final_format = resImg
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

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# np.savetxt('dataset.txt',dataset , fmt='%10.5f')
# dataset.tofile('dataset.dat')
# np.save('data.npy',dataset)
save_obj(dataset,'data')
