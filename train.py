import argparse
import logging
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
# import cv2

from p3d_model import transfer_model, P3D199, get_optim_policies

logging.basicConfig(
	format='%(asctime)s %(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Pesudo 3D on UCF101 Dataset")

parser.add_argument("--load", 
					help="Load saved network weights. (default = best_weights)")
parser.add_argument("--save", 
					help="Save network weights. (default = cnn_weight)")  
parser.add_argument("--epochs", default=20, type=int,
					help="Epochs through the data. (default=20)")  
parser.add_argument("--learning_rate", "-lr", default=1e-2, type=float,
					help="Learning rate of the optimization. (default=0.1)")
parser.add_argument("--estop", default=1e-2, type=float,
					help="Early stopping criteria on the development set. (default=1e-2)")               
parser.add_argument("--batch_size", default=10, type=int,
					help="Batch size for training. (default=10)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
					help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[], nargs='+', type=str,
					help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--size", default=160, type=int,
					help="size of images.")
parser.add_argument("--split", default=1, type=int,
					help="the number of train-test split of ucf101")
parser.add_argument("--machine", default='ye_home', type=str,
					help="which machine to run the code. choice from ye_home and marcc")

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
		for i in range(num_frame):
			img = Image.open(images[i])
			img = img.convert('RGB')
			# img = self.transform(img)
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




def main(options):
	# Path configuration


	machine =options.machine
	split = options.split

	# Path to the directories of features and labels
	if machine == 'ye_home':
		train_file = '/home/ye/Works/C3D-TCN-Keras/ucfTrainTestlist/trainlist0'+str(split)+'.txt'
		test_file = '/home/ye/Works/C3D-TCN-Keras/ucfTrainTestlist/testlist0'+str(split)+'.txt'
		data_folder = '/home/ye/Works/C3D-TCN-Keras/frames'
		label_file = '/home/ye/Works/C3D-TCN-Keras/ucfTrainTestlist/classInd.txt'
	elif machine == 'marcc':
		#train_file = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/ucfTrainTestlist/trainlist0'+str(split)+'.txt'
		#test_file = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/ucfTrainTestlist/testlist0'+str(split)+'.txt'
		#data_folder = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/frames'
		#label_file = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/ucfTrainTestlist/classInd.txt'
		
		train_file = './ucfTrainTestlist/trainlist0'+str(split)+'.txt'
		test_file = './ucfTrainTestlist/testlist0'+str(split)+'.txt'
		data_folder = './frames'
		label_file = './ucfTrainTestlist/classInd.txt'



	

	

	transformations = transforms.Compose([transforms.Scale((options.size,options.size)),
									transforms.ToTensor(),
									transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
									])
	
	dset_train = ucf101Dataset(data_folder, train_file, label_file, transformations, size=options.size)

	dset_test = ucf101Dataset(data_folder, test_file, label_file, transformations, size=options.size)


	train_loader = DataLoader(dset_train,
							  batch_size = options.batch_size,
							  shuffle = True,
							 )

	test_loader = DataLoader(dset_test,
							 batch_size = options.batch_size,
							 shuffle = False,
							 )

	use_cuda = (len(options.gpuid) >= 1)
	#if options.gpuid:
		#cuda.set_device(int(options.gpuid[0]))
	
	# Initial the model
	model = P3D199(pretrained=True,num_classes=400)
	model = transfer_model(model,num_classes=101)


	if use_cuda > 0:
		model.cuda()


	# Binary cross-entropy loss
	criterion = torch.nn.CrossEntropyLoss()
	# criterion = torch.nn.NLLLoss()
	# optimizer = eval("torch.optim." + options.optimizer)(model.parameters())get_optim_policies(model=None,modality='RGB',enable_pbn=True)
	optimizer = eval("torch.optim." + options.optimizer)(get_optim_policies(model=model,modality='RGB',enable_pbn=True))

	# main training loop
	last_dev_avg_loss = float("inf")
	for epoch_i in range(options.epochs):
		logging.info("At {0}-th epoch.".format(epoch_i))
		train_loss = 0.0
		correct = 0.0
		for it, train_data in enumerate(train_loader, 0):
			vid_tensor, labels = train_data
			if use_cuda:
				vid_tensor, labels = Variable(vid_tensor).cuda(),  Variable(labels).cuda()
			else:
				vid_tensor, labels = Variable(vid_tensor), Variable(labels)

			train_output = model(vid_tensor)
			train_output = torch.nn.LogSoftmax()(train_output)

			# print 'model output shape: ', train_output.size(), ' | label shape: ', labels.size()
			# print (train_output.size())

			loss = criterion(train_output, labels)
			train_loss += loss.data[0]

			pred = train_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

			logging.info("loss at batch {0}: {1}".format(it, loss.data[0]))
			# logging.debug("loss at batch {0}: {1}".format(it, loss.data[0]))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if it % 50 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch_i, it * len(vid_tensor), len(train_loader.dataset),
					100. * it / len(train_loader), loss.data[0]))

			
		train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
		training_accuracy = (correct / len(dset_train))
		logging.info("Average training loss value per instance is {0} at the end of epoch {1}".format(train_avg_loss, epoch_i))
		logging.info("Training accuracy is {0} at the end of epoch {1}".format(training_accuracy, epoch_i))


if __name__ == "__main__":
	ret = parser.parse_known_args()
	options = ret[0]
	if ret[1]:
		logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
	main(options)
