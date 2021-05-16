import os
import torch
import pickle
import fnmatch
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class shapenet_views_dataset(Dataset):
	def __init__(self, root_dir, near_far_size, num_views=5, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.near_far_size = torch.Tensor(near_far_size)
		self.length = sum([len(fnmatch.filter(files, '*.png')) for r, d, files in os.walk(self.root_dir)])//num_views
		self.file_list = np.concatenate([np.array([os.path.join(r, mfile) for mfile in fnmatch.filter(files, '*.png')]) 
							for r, d, files in os.walk(self.root_dir)]).flatten()
		self.file_list = natsorted(self.file_list)
		temp_camera_dict = pickle.load(open(os.path.join(root_dir, 'camera_matrices.pkl'), 'rb'))
		self.camera_intrinsics = temp_camera_dict['intrinsic_matrices']
		self.camera_extrinsics = temp_camera_dict['extrinsic_matrices']
		print(self.camera_extrinsics.shape)
		self.num_views = num_views

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		idx_start = idx*self.num_views
		label = self.file_list[idx_start].split("/")[-2]
		img = [torch.from_numpy(plt.imread(self.file_list[idx_start+i])) for i in range(self.num_views)]
		img = torch.stack(img)
		if self.transform:
			img = self.transform(img)
		#Temp stuff
		img = img.reshape(1, 5, 4, 256, 256)
		return img, self.camera_intrinsics, self.camera_extrinsics, self.near_far_size, label