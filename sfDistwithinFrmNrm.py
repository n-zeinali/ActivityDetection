from sfSetting import SaveFileSetting as sfSetting
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
from scipy.spatial import distance
from joblib import Parallel, delayed
import multiprocessing

def all_dist_within_frm(org_val_skl):
		num_cores = multiprocessing.cpu_count()
		# print(num_cores)
		# num_cores = 1
		h, width, length = org_val_skl.shape
		frm_len = range(length)
		disMat = Parallel(n_jobs=num_cores, backend="threading")(delayed(dist_frm)(l,org_val_skl,h,width,length) for l in frm_len)

		disMat = np.array(disMat)
		# print(disMat)
		# for l in range(length):
		# 	dist_frm(l,org_val_skl,h,width,length)
		# disMat = dist_frm(org_val_skl)
		disMat = np.rollaxis(disMat, 0, 2)
		# print(disMat.shape)
		return disMat


def dist_frm(l,org_val_skl,h,width,length):
    # print(l)
    disMat = np.zeros([h, int(width*(width-1)/2), length])
    # for l in range(length):
    counter = 0
    frm = org_val_skl[:, :, l]
    disMat = np.zeros([int(width*(width-1)/2)])
    for w1 in range(width):
        for w2 in range(w1):
            a = org_val_skl[:, w1, l]
            b = org_val_skl[:, w2, l]
            # c = np.sqrt(np.sum((a-b)**2))
            c = distance.euclidean(a, b)
            disMat[counter] = c
            counter = counter + 1
        # print(counter)
        # print(width*(width-1)/2)
    return disMat

# g_cnt = 0;
def parse_files(setting, file, path, rootDr):
	 # global g_cnt
	 # g_cnt = g_cnt + 1
	 # print(g_cnt)
	 print(rootDr)
	 print('============================================')
	 mat = scipy.io.loadmat(path+'//'+file)
	 # print(mat.shape)
	 # print mat
	 data = mat['samples']

	 val = data[0]
	 # print(type(val))
	 if( val != -1):
		 val_skl = val['skeleton'][0]
		 # org_val_skl = val['originalSkeleto'][0]
		 # print(val_skl.shape)
		 more_info = {
				 'act_clss': str(val['actionClass'][0][0, 0]),
				 'setup_number': str(val['setupNumber'][0][0, 0]),
				 'camera_id': str(val['cameraID'][0][0, 0]),
				 'performer': str(val['performerID'][0][0, 0]),
				 'replication_number': str(val['replicationNumber'][0][0, 0])
		 }
		 print('class', more_info['act_clss'])
		 # sys.exit()
		 pos, joint, frame = val_skl.shape
		 #print('skeleton', val_skl.shape)       
		 
		 val_skl_np = np.array(val_skl)
		 # print(final_format.shape)
		 # org_val_skl = np.array(org_val_skl)
		 final_format = all_dist_within_frm(val_skl_np)
		 data_st = np.array([final_format, more_info])
		 file_path = setting.getPath(more_info)
		
		 #You can save three format files such as "pkl", "npz" and "mat"
		 setting.save_obj(data_st,file_path, "pkl")

		 # filesCounter = filesCounter+1;
setting = sfSetting('allDistWithinFrmNrm')
start_time = setting.startTime()
files = os.listdir(setting.path)

# for file in files:
# 	parse_files(file,path,rootDr)

num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(parse_files)(setting, file, setting.path, setting.rootDr) for file in files)




