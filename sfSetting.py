import numpy as np
import scipy.io
import pickle
import shutil
import time
import sys
import os



class SaveFileSetting(object):
	"""docstring for SfSetting"""
	#source to read mat files
	path  = "/media/ryz/983ADA463ADA2154/Datasets/NTU/NTUMatData"
	#destination: path that file save
	rootDr_path = '/home/share/NTUFeatures/'
	rootDr= ''

	start_time = 0
	
	def __init__(self, dirname):	
		self.rootDr = os.path.dirname(self.rootDr_path+ dirname+ '/')
		# Attention: remove last folder then make new files!!!!
		if os.path.exists(self.rootDr):
			shutil.rmtree(self.rootDr)
		os.makedirs(self.rootDr)
		os.makedirs(self.rootDr+'/train')
		os.makedirs(self.rootDr+'/test')

	def save_obj(self, obj, name, format="pkl"):
		#with open(name + '.pkl', 'wb+') as f:
				#pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

		if(format == "mat"):
			scipy.io.savemat(name+'.mat', {'vect':obj}, do_compression=True)
		elif(format == "npz"):
			np.savez_compressed(name, a=obj)
		else:
			with open(name + '.pkl', 'wb+') as f:
				pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


		#dict = {'mat': self.saveMat, 'npz': self.saveNpz, 'pkl': self.savePkl}
		#try:
		    #dict[format](obj, name)
		#except KeyError:
			#print('Error : File\'s format should be either [mat / npz / pkl]')
    

	def load_obj(self, name ):
			with open( name + '.pkl', 'rb') as f:
					return pickle.load(f)

	def getPath(self, more_info):
		name_folder = 'test' if more_info['camera_id'] == '1' else 'train' 

		file_name = 'S'+more_info['setup_number'].zfill(3)+'C'+more_info['camera_id'].zfill(3)+'P'+more_info['performer'].zfill(3)+'R'+more_info['replication_number'].zfill(3)+'A'+more_info['act_clss'].zfill(3)
		file_path = self.rootDr +'/'+ name_folder+ '/'+ file_name
		
		return file_path
		
	def savePkl(self, obj, name):
		with open(name + '.pkl', 'wb+') as f:
				pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
	def saveMat(self, obj, name):
		scipy.io.savemat(name+'.mat', {'vect':obj}, do_compression=True)
	def saveNpz(self, obj, name):
		np.savez_compressed(name, a=obj)

	def startTime(self):
		self.start_time = time.time()

	def timeElapsed(self):
		time_elapsed = time.time() - self.start_time
		m, s = divmod(time_elapsed, 60)
		h, m = divmod(m, 60)
		print("--- %d:%02d:%02d ---" % (h, m, s))
        #print(time_elapsed)

    


		 

