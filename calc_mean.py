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


def save_obj(obj, name, format="pkl"):
	if(format == "mat"):
		scipy.io.savemat(name+'.mat', {'vect': obj}, do_compression=True)
	elif(format == "npz"):
		np.savez_compressed(name, a=obj)
	else:
		with open(name + '.pkl', 'wb+') as f:
			pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, format="pkl"):
	if(format == "mat"):
		scipy.io.loadmat(name)
	elif(format == "npz"):
		npzfile = np.load(name)
		result = npzfile['a']
		return result[0], result[1]
	else:
		with open(name, 'rb') as f:
			return pickle.load(f)


rootDr = os.path.dirname('dataset/')
files = os.listdir(os.path.join(rootDr, 'train'))

# mean = load_obj('mean.pkl', format="pkl")
# h, w  = mean.shape
# temp = np.zeros([h,w,1])
# temp[:,:,0] = mean
# mean = temp


# print(mean.shape)
# sys.exit() 

counter = 0
# print(len(files))
pkl_path = os.path.join(rootDr, 'train',files[0])
sample, category = load_obj(pkl_path, "pkl")
h, w, d  = sample.shape
# total_sum = np.zeros([h, w, len(files)])
sm = np.zeros([h, w])
for idx,file in enumerate(files):

	print('============================================')

	print('Number of files', idx)
	pkl_path = os.path.join(rootDr, 'train',file)
	# print(rootDr)
	# print(file)
	# print(pkl_path)

	sample, category = load_obj(pkl_path, "pkl")
	h, w, d  = sample.shape
	# sample = sample - np.repeat(mean,d,axis = 2)


	# if (counter == 0):
	# 	h, w, d = sample.shape
	# 	total_sum = np.zeros([h, w, len(files)])
		# print(total_sum.shape)
		# sys.exit()
	# print(sample)
	sm = sm + np.sum(sample, axis=2)

	# print(sm.shape)
	# print(total_sum.shape)
	# temp_sum = np.zeros([h,w,1])
	# temp_sum[:,:,0] = sm
	# total_sum[:,:,counter] = sm
	# print(sm)
	# print(total_sum.shape)
	# sys.exit()
	counter = counter + d

# mean = np.mean(total_sum,axis = 2)
mean = sm / counter
print(mean.shape)
print(mean)
save_obj(mean,'mean', "pkl")
