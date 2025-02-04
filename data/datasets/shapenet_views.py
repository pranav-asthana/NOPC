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
	def __init__(self, root_dir, near_far_size, is_train=True, num_views=20, transform=None):
		print('DEBUG: dataset root directory: "{}"'.format(root_dir))
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

		# hard coded because the test-train split is difficult to perform via an algorithm
		self.batch_size = 5

		# train-test split ratio is 15:5 for 20 images
		self.is_train = is_train
		train_sample_size = int((num_views/4)*3)
		test_sample_size = int((num_views/4))
		self.samp_size = train_sample_size if is_train else test_sample_size
		
		full_sampling_indices = np.arange(num_views)
		temp_sampling_indices = np.linspace(0, num_views-1, train_sample_size, dtype=np.float).astype(np.int32)
		self.sampling_indexes = (temp_sampling_indices if is_train else 
								np.setdiff1d(full_sampling_indices, temp_sampling_indices))
		if is_train is False: self.sampling_indexes[0] = 0

		# print('DEBUG: full_sampling_indices: {}'.format(full_sampling_indices))
		# print('DEBUG: temp_sampling_indices: {}'.format(temp_sampling_indices))
		# self.sampling_indexes = torch.randperm(num_views)[:self.samp_size]
		# for eidx, sidx in enumerate(self.sampling_indexes):
		# 	if (eidx != 0) and (sidx == 0): 
		# 		self.sampling_indexes[eidx] = self.sampling_indexes[0]
		# 		self.sampling_indexes[0] = 0 
		# print('DEBUG: self.sampling_indexes: {}'.format(self.sampling_indexes))

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		idx_start = idx*self.num_views
		label = self.file_list[idx_start].split("/")[-2]

		cam_extr = None
		img = None
		if self.is_train:
			views_to_sample = np.random.permutation(self.sampling_indexes[1:])[:self.batch_size]
			views_to_sample[0] = 0
			# print('DEBUG: views_to_sample: {}'.format(views_to_sample))
			img = torch.stack([torch.from_numpy(plt.imread(self.file_list[idx_start+i])) for i in views_to_sample])
			cam_extr = torch.stack([self.camera_extrinsics[sidx] for sidx in views_to_sample])
			# img = torch.stack([img[sidx] for sidx in views_to_sample])
		else:
			img = torch.stack([torch.from_numpy(plt.imread(self.file_list[idx_start+i])) for i in self.sampling_indexes])
			cam_extr = torch.stack([self.camera_extrinsics[sidx] for sidx in self.sampling_indexes])
			# img = torch.stack([img[sidx] for sidx in self.sampling_indexes])
		
		if self.transform:
			img = self.transform(img)

		# print(img.shape)
		img = img.permute(0, 3, 1, 2)
		cam_intrs = self.camera_intrinsics[:self.batch_size, ...]
		return img, cam_intrs, cam_extr, self.near_far_size, label