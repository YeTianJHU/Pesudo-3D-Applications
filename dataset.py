import os
from os import listdir
from os.path import isfile, join, isdir

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

#import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
from torchvision.transforms import ToPILImage


class ucf101Dataset(Dataset):
	def __init__(self, data_folder, split_file, label_file, transform, num_labels=101, num_frame=16, channel=3, size=160):

		self.data_folder = data_folder
		self.transform = transform
		self.video_lst = self.read_split_file(split_file)
		self.num_frame = num_frame
		self.channel = channel
		self.size = size
		self.label_dict = self.trans_label(label_file)
		self.num_labels = num_labels

		
	def __getitem__(self, index):
		# try:
		video_name_get =  self.video_lst[index][1]
		video_name = video_name_get.split('.')[0]
		# print video_name
		video_path = os.path.join(self.data_folder, video_name)

		video_tensor = self.get_video_tensor(video_path, self.num_frame, self.channel, self.size)

		label_name = self.video_lst[index][0]
		label_idx = self.label_dict[label_name] -1  # -1 because we need to get one-hot

		# one_hot_label = torch.LongTensor(self.num_labels)
		# one_hot_label = torch.FloatTensor(self.num_labels)
		# one_hot_label[label_idx] = 1
		one_hot_label = label_idx

		return video_tensor, one_hot_label
		# except:
		# 	return torch.FloatTensor(self.channel,self.num_frame,self.size,self.size), torch.FloatTensor(self.num_labels)

	def __len__(self):
		return len(self.video_lst)

	def read_split_file(self, dir):
		res = []
		with open(dir) as f:
			for line in f:
				split = line.split('/')
				res.append(split)
		return res

	def collect_files(self, dir_name, file_ext=".jpg", sort_files=True):
		allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

		these_files = []
		for i in range(0,len(allfiles)):
			_, ext = os.path.splitext(os.path.basename(allfiles[i]))
			if ext == file_ext:
				these_files.append(allfiles[i])

		if sort_files and len(these_files) > 0:
			these_files = sorted(these_files)

		return these_files

	def get_video_tensor(self, dir, num_frame, channel, size):
		images = self.collect_files(dir)
		flow = torch.FloatTensor(channel,num_frame,size,size)
		seed = np.random.random_integers(0,len(images)-num_frame) #random sampling
		for i in range(num_frame):
			img = Image.open(images[i+seed])
			img = img.convert('RGB')
			img = self.transform(img)
			flow[:,i,:,:] = img
		return flow

	def trans_label(self, txt):
		label_list = np.genfromtxt(txt, delimiter=' ', dtype=None)
		label_dict = {}
		num_lab = 101
		for i in range(num_lab):
			label_dict[label_list[i][1].astype(str)] = label_list[i][0] 
		return label_dict