import _pickle as cPickle
import numpy as np
import scipy.io
import zipfile
import tarfile
import pickle
import shutil
import os
import lzma
import zlib
import h5py
import cv2
import sys
import math
from scipy.spatial import distance
from joblib import Parallel, delayed
import multiprocessing
from sfSetting import SaveFileSetting as sfSetting
import time



# g_cnt = 0;
def parse_files(setting, file, path, rootDr):
	 # global g_cnt
	 # g_cnt = g_cnt + 1
	 # print(g_cnt)
	 # print(rootDr)
	 # print('============================================')
	 mat = scipy.io.loadmat(path+'//'+file)
	 # print(mat.shape)
	 # print mat
	 data = mat['samples']

	 val = data[0]
	 # print(type(val))
	 if( val != -1):
		 # val_skl = val['skeleton'][0]
		 org_val_skl = val['originalSkeleto'][0]
		 # print(val_skl.shape)
		 more_info = {
				 'act_clss': str(val['actionClass'][0][0, 0]),
				 'setup_number': str(val['setupNumber'][0][0, 0]),
				 'camera_id': str(val['cameraID'][0][0, 0]),
				 'performer': str(val['performerID'][0][0, 0]),
				 'replication_number': str(val['replicationNumber'][0][0, 0])
		 }

		 final_format = np.array(org_val_skl)
		 data_st = np.array([final_format, more_info])
		 #print(more_info['camera_id'])
		 #print(file_path)
		 file_path = setting.getPath(more_info)
		
		 #You can save three format files such as "pkl", "npz" and "mat"
		 setting.save_obj(data_st,file_path, "pkl")
		 # filesCounter = filesCounter+1;


setting = sfSetting('Org')
start_time = setting.startTime()
files = os.listdir(setting.path)

counter = 1;
percent = 0;
# print("progress =: ",percent)/
# print((counter) / len(files))
# sys.exit()
for file in files:
	# print("progress =: ",int(np.round((counter / len(files))*100)),'%')
	if np.round((100*counter / len(files))) != percent :
		percent = np.round(100*counter / len(files))
		print("progress =: ",int(np.round(percent)),'%')
	parse_files(setting, file, setting.path, setting.rootDr)
	counter = counter + 1
	# print(counter)
setting.timeElapsed()
# num_cores = multiprocessing.cpu_count()
# Parallel(n_jobs=num_cores)(delayed(parse_files)(file,path,rootDr) for file in files)




