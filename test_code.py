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

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

from torchvision.transforms import ToPILImage

def read_split_file(dir):
    res = []
    with open(dir) as f:
        for line in f:
            split = line.split('/')
            res.append(split)
    return res

def get_video_tensor(dir):
	images = collect_files(dir)
	flow = torch.FloatTensor(3,16,160,160)
	seed = np.random.random_integers(0,len(images)-16) #random sampling
	print seed, len(images)
	for i in range(16):
		img = Image.open(images[i+seed])
		img = img.convert('RGB')
		# img = self.transform(img)
		img = transformations(img)
		flow[:,i,:,:] = img

	to_pil_image = ToPILImage()
	img = to_pil_image(flow[:,7,:,:])
	img.show()


transformations = transforms.Compose([transforms.Scale((160,160)),
								transforms.ToTensor()
								])


def collect_files(dir_name, file_ext=".jpg", sort_files=True):
	allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

	these_files = []
	for i in range(0,len(allfiles)):
		_, ext = os.path.splitext(os.path.basename(allfiles[i]))
		if ext == file_ext:
			these_files.append(allfiles[i])

	if sort_files and len(these_files) > 0:
		these_files = sorted(these_files)

	return these_files

def trans_label(txt):
	# label_list = np.loadtxt(txt)
	label_list = np.genfromtxt(txt, delimiter=' ', dtype=None)
	label_dict = {}
	# num_lab = len(label_list)
	num_lab = 101
	for i in range(num_lab):
		print (label_list[i][1].astype(str), label_list[i][0])
		label_dict[label_list[i][1].astype(str)] = label_list[i][0]
	# print label_dict['HandStandPushups']
	return label_dict

# file = os.path.join('/media/ye/youtube-8/ucfTrainTestlist/testlist01.txt')
# res = read_split_file(file)
# print res[0][1][:-6]

dir = os.path.join('/home/ye/Works/C3D-TCN-Keras/frames/v_ApplyEyeMakeup_g01_c02')
get_video_tensor(dir)

# label_dict = trans_label('./ucfTrainTestlist/classInd.txt')
# print (label_dict)
# label = label_dict['Swing']
# print(label)
