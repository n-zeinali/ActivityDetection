import _pickle as cPickle
import numpy as np
import scipy.io
import zipfile
import tarfile
import pickle
import shutil
import lzma
import zlib
import h5py
import cv2
import sys
import math
import os



def save_obj1(obj, name ):
    with tarfile.ZipFile(name + '.zip', 'w', zipfile.ZIP_DEFLATED) as zf:  
        #zf = zipfile.ZipFile('zipped_pickle.zip', 'wb+', zipfile.ZIP_DEFLATED)
        #zf.writestr('data.pkl', cPickle.dumps(some_data, -1))
        print(obj)
        print()
        data_out = zlib.compressobj(obj)
        print(data_out)
        print('...de....')
        print(zlib.decompressobj(data_out))
        sys.exit()
        zf.writestr('data.pkl', cPickle.dumps(data_out, pickle.HIGHEST_PROTOCOL))

def save_obj(obj, name, format="pkl"):
  if(format == "mat"):
    scipy.io.savemat(name+'.mat', {'vect':obj}, do_compression=True)
  elif(format == "npz"):
    np.savez_compressed(name, a=obj)
  else:
    with open(name + '.pkl', 'wb+') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

trainFlag = '--train' in sys.argv
cross_view = '--view' in sys.argv
diffFlag = '--diff' in sys.argv
hieracFlag = '--hierac' in sys.argv
originalFlag = '--origin' in sys.argv
origDiffFlag = '--originDiff' in sys.argv
#print(trainFlag)

  

# if trainFlag:
#   #destination: path that file save
#   rootDr = os.path.dirname('dataNp/train/')
#   #read mat file  
#   path  = "data/train"
# else:
#   #destination: path that file save
#   rootDr = os.path.dirname('dataNp/test/')
#   #read mat file  
#   path  = "/media/deepface/983ADA463ADA2154/Datasets/NTU/NTUMatData"



# if cross_view:
  #read mat file  
path  = "/media/deepface/983ADA463ADA2154/Datasets/NTU/NTUMatData"
  #destination: path that file save
rootDr = os.path.dirname('dataset2/')



# Attention: remove last folder then make new files!!!!
# print(rootDr)
if os.path.exists(rootDr):
  shutil.rmtree(rootDr) 
  os.makedirs(rootDr)
  os.makedirs(rootDr+'/train')
  os.makedirs(rootDr+'/test')
else:
  os.makedirs(rootDr)
  os.makedirs(rootDr+'/train')
  os.makedirs(rootDr+'/test')


files = os.listdir(path)
counter = 0
filesCounter =0;
skipped_count = 0

for file in files:
   print('============================================')
   counter = counter + 1
   print('Number of files', counter)
   # print('Name of file' + str(path+'/'+file))
   # file = 'S010C001P021R001A060.mat'
   #reead matlab file
   mat = scipy.io.loadmat(path+'//'+file)
   # print(mat.shape)
   # print mat
   data = mat['samples']
   # row, smpls = data.shape
   # sampels = {}
   # print('shape', data.shape)
   

   # print data
   
   # for smpl in range(smpls):
       #print('smpl', smpl)
       # val = data[smpl]
   val = data[0]
   # print(type(val))
   if( val == -1):
       # print('catched')
       skipped_count = skipped_count + 1
       continue

   #print(type(val[0, 0]))
   #print(len(val[0, 0]))
   #print('============================================')
   val_skl = val['skeleton'][0]
   org_val_skl = val['originalSkeleto'][0]
   # print(val_skl.shape)
   more_info = {
       'act_clss': str(val['actionClass'][0][0, 0]),
       'setup_number': str(val['setupNumber'][0][0, 0]),
       'camera_id': str(val['cameraID'][0][0, 0]),
       'performer': str(val['performerID'][0][0, 0]),
       'replication_number': str(val['replicationNumber'][0][0, 0])
   }
   if int(more_info['act_clss']) <= 49:

     print('class', more_info['act_clss'])
     # sys.exit()
     pos, joint, frame = val_skl.shape
     #print('skeleton', val_skl.shape)       
     
     final_format = np.array(val_skl)
     # print(final_format.shape)
     
     # img = np.zeros([w,d,h])
     
     #make interpolation and resize
     #img[:,:,0] = final_format[0,:,:]
     #img[:,:,1] = final_format[1,:,:]
     #img[:,:,2] = final_format[2,:,:]
     #res = cv2.resize(img,(224,224),fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
     #resImg = np.zeros([h,224,224])
     #resImg[0,:,:] = res[:,:,0]       
     #resImg[1,:,:] = res[:,:,1]  
     #resImg[2,:,:] = res[:,:,2]  
     #end interpolation
     if originalFlag:
      # print('1: ',final_format.shape)
      final_format = np.append(final_format,np.array(org_val_skl),axis = 1)
      # print('2: ',final_format.shape)

     # save smapel
     #print(final_format.shape)


     if hieracFlag:
        sample =final_format
        h,w,d = sample.shape
        # img = np.zeros([w,d,h])
        # levels = int(math.log(d,2))

        step = 2
        res = sample
        
        # print('sample[:,:,1] shape',sample[:,:,1].shape)
        while step < int(d / 2):
          colLists = [] 
          for l in range(0,d,step):
            colLists = np.append(colLists,int(l))
            # print(type(colLists))
            colLists =colLists.astype(int)
            # print(type(colLists))
            
            # temp = np.zeros([h,w,1])

            
          # print(colLists)
          temp  = sample[:,:,colLists]
          # print(temp.shape)
          res = np.append(res,temp,axis = 2)
          step = step * 2
          # print('res shape : ',res.shape)

        final_format = res
        # print('final_format shape :',final_format.shape)
        # sys.exit()
        # print('img shape : ',sample.shape)

        #make interpolation and resize
        # img[:,:,0] = sample[0,:,:]
        # img[:,:,1] = sample[1,:,:]
        # img[:,:,2] = sample[2,:,:]
        # # print(img.shape)
        # res = img
        # print(res.shape)
        # for l in range(0,levels-2):
        #   # print(cv2.resize(img,(w,int(round(d/2))),fx=0, fy=0, interpolation = cv2.INTER_NEAREST).shape)
        #   res = np.append(res,cv2.resize(img,(int(round(d/2)),w),fx=0, fy=0, interpolation = cv2.INTER_NEAREST),axis = 1)
        #   d = round(d / 2)
        #   # print(res.shape)

        # sys.exit()
        # h,w,d = res.shape
        # # print(res.shape)
        # resImg = np.zeros([d,h,w])
        # resImg[0,:,:] = res[:,:,0]       
        # resImg[1,:,:] = res[:,:,1]  
        # resImg[2,:,:] = res[:,:,2]  
       # print(diff.shape)
     # sys.exit()

     if diffFlag:
       h,w,d = final_format.shape
       if origDiffFlag:
        w = w *2
        diff = np.diff(np.append(final_format,np.array(org_val_skl),axis = 1))
       else:
        diff = np.diff(final_format)
       
       zeroCol = np.zeros([h,w,1])
       # final_format = diff + zeroCol
       # print(zeroCol.shape)
       diff = np.append(diff, zeroCol, axis=2)
       final_format = np.append(final_format, diff, axis=1)
       # print(final_format.shape)


     print('final_format shape : ',final_format.shape)
     data_st = np.array([final_format, more_info['act_clss']])
     name_folder = ''
     # if cross_view:
      # if int(data_st)<49:
     if (more_info['camera_id'] == '1'):
      name_folder = 'test/'
     else:
      name_folder = 'train/'
     #S001C002P003R002A013
     file_name = 'S'+more_info['setup_number'].zfill(3)+'C'+more_info['camera_id'].zfill(3)+'P'+more_info['performer'].zfill(3)+'R'+more_info['replication_number'].zfill(3)+'A'+more_info['act_clss'].zfill(3)
     file_path = rootDr +'/'+ name_folder + file_name
     # print(file_path)
    
     #you save three format files such as "pkl", "npz" and "mat"
     print(file_path)
     save_obj(data_st,file_path, "pkl")
     filesCounter = filesCounter+1;




