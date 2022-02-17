import _pickle as cPickle
import numpy as np
import zipfile
import pickle
import torch
import lzma
import cv2
import sys
import re
import os

def load_obj(name, format="pkl"):
  if(format == "mat"):
    scipy.io.loadmat(name)
  elif(format == "npz"):
        npzfile = np.load(name)
        result  = npzfile['a']
        return result[0], result[1]
  else:
    with open( name , 'rb') as f:
        return pickle.load(f)        
    

class CustomDataset(torch.utils.data.Dataset):
    '''
        Custom Dataset object for the CDiscount competition
        Parameters:
            root_dir - directory including category folders with images

        Example:
        test/
            `1.pkl
             2.pkl
                ...
        train/
            1.pkl
            2.pkl
            ...
    '''
    directories = ['allDistWithinFrm', 'allDistWithinFrmNrm', 'diffNrm', 'diffOrg', 'HieracNrm', 'Nrm', 'Org']
    phase = ''
    path  = ''

    
    def __init__(self, root_dir, phase, transform=None):
        #first using
        self.root_dir = root_dir + "/"+ self.directories[6]+ "/"+ phase
        self.phase = phase
        self.path = root_dir
        
        #print(self.root_dir)
        #sys.exit()
  
        self.files = []
        for (dirpath, dirnames, filenames) in os.walk(self.root_dir):
            for f in filenames:
                if f.endswith(('.npz', '.pkl', '.mat')):
                    o = {}
                    o['pkl_path'] = f
                    self.files.append(o)
        self.transform = transform
        
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        pkl_path = self.root_dir + '/'+ self.files[idx]['pkl_path']
        # category = self.files[idx]['category']

        #load three format files such as "pkl", "npz" and "mat"

        sample ,category =  load_obj(pkl_path, "pkl")
        #Concat type of feature from origin data
        final_format = sample
        # load other features. Index features gets from name of dirctories variable. 
        features_used  = ['diffNrm']
        #concat_feature = np.array() 
        for item in features_used:
            feature_path = self.path + '/'+ item+ '/'+ self.phase+ '/'+ self.files[idx]['pkl_path']
            feature_smpl, feature_cat = load_obj(feature_path, "pkl")
            #print(feature_path)
            #print(feature_smpl.shape)
            
            final_format = np.append(final_format, feature_smpl, axis=1)


        h,w,d = final_format.shape
        img = np.zeros([w,d,h])
       
        #make interpolation and resize
        img[:,:,0] = final_format[0,:,:]
        img[:,:,1] = final_format[1,:,:]
        img[:,:,2] = final_format[2,:,:]
        res = cv2.resize(img,(224, 224),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
        resImg = np.zeros([h,224, 224])
        resImg[0,:,:] = res[:,:,0]       
        resImg[1,:,:] = res[:,:,1]  
        resImg[2,:,:] = res[:,:,2]  
        #end interpolation
        sample = torch.from_numpy(np.array(resImg)).float()
        # category = torch.from_numpy(np.array(category))
        
        # if self.transform:
        #     # print('================ToTensor================')
        #     sample = self.transform(sample)
 
        # print('after transform: ',sample.shape)
        #print(category)
        #sys.exit()
        return {'sample': sample, 'category': int(category['act_clss']+'')}
