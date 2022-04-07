import os
import re
import sys
import time
import random
import logging
import argparse
import numpy as np
# from PIL import Image
from scipy import misc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import dataset as dataset
from arguments import Args
from model import Srnet
opt = Args()

logging.basicConfig(filename='training.log',format='%(asctime)s %(message)s', level=logging.DEBUG)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# To find latest checkpoint. From where to begin the training
def latest_checkpoint():
	if os.path.exists(opt.checkpoints_dir):
		all_chkpts = "".join(os.listdir(opt.checkpoints_dir))
		latest = max(map(int, re.findall('\d+', all_chkpts)))
	else:
		latest = None
	return latest

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decays by 10 every 30 epochs"""
	lr = opt.lr * (0.1 ** (epoch // 80))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

# Weight initialization for conv layers and fc layers
def weights_init(m):
	if isinstance(m, nn.Conv2d):
		torch.nn.init.xavier_uniform(m.weight.data)
		if m.bias is not None:
			torch.nn.init.constant_(m.bias.data, 0.2)
	elif isinstance(m, nn.Linear):
		torch.nn.init.normal_(m.weight.data, mean=0., std=0.01)
		torch.nn.init.constant_(m.bias.data, 0.)


if __name__ == '__main__':

	train_data = dataset.Dataset_Load(opt.cover_path, opt.stego_path, opt.train_size,
									transform= transforms.Compose([
										dataset.ToPILImage(),
										dataset.RandomRotation(p=0.5),
										dataset.ToTensor()]))

	val_data = dataset.Dataset_Load(opt.valid_cover_path, opt.valid_stego_path, opt.val_size,
									transform=dataset.ToTensor())

	train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
	valid_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False)

# model creation and initialization
	model = Srnet()
	model.to(device)
	model = model.apply(weights_init)

	loss_fn = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adamax(model.parameters(), lr = opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
	
	check_point = latest_checkpoint()
	if check_point == None:
		st_epoch = 1
		if not os.path.exists(opt.checkpoints_dir):
			os.makedirs(opt.checkpoints_dir)
		print("No checkpoints found!!, Retraining started... ")
	else:
		pth = opt.checkpoints_dir + 'net_' + str(check_point) +'.pt'
		ckpt = torch.load(pth)
		st_epoch = ckpt['epoch'] + 1
		model.load_state_dict(ckpt['model_state_dict'])
		optimizer.load_state_dict(ckpt['optimizer_state_dict'])

		print("Model Loaded from epoch " + str(st_epoch) + "..")

	for epoch in range(st_epoch, opt.num_epochs+1):
		training_loss = []
		training_accuracy = []
		validation_loss = []
		validation_accuracy = []
		test_accuracy = []

		model.train()
		st_time = time.time()
		adjust_learning_rate(optimizer, epoch)

		for i, train_batch in enumerate(train_loader):
			images = torch.cat((train_batch['cover'], train_batch['stego']), 0)
			labels = torch.cat((train_batch['label'][0], train_batch['label'][1]), 0)

			images = images.to(device, dtype=torch.float)
			labels = labels.to(device, dtype=torch.long)

			optimizer.zero_grad()

			outputs = model(images)

			loss = loss_fn(outputs, labels)
			loss.backward()

			optimizer.step()

			training_loss.append(loss.item())

			prediction = outputs.data.max(1)[1]
			accuracy = prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
			training_accuracy.append(accuracy.item())

			sys.stdout.write('\r Epoch:[%d/%d] Batch:[%d/%d] Loss:[%.4f] Acc:[%.2f] lr:[%.4f]'
				%(epoch, opt.num_epochs, i+1, len(train_loader), training_loss[-1], training_accuracy[-1], optimizer.param_groups[0]['lr']))
		
		end_time = time.time()
		
		model.eval()
		with torch.no_grad():

			for i, val_batch in enumerate(valid_loader):
				images = torch.cat((val_batch['cover'], val_batch['stego']),0)
				labels = torch.cat((val_batch['label'][0], val_batch['label'][1]),0)

				images = images.to(device, dtype=torch.float)
				labels = labels.to(device, dtype=torch.long)

				outputs = model(images)

				loss = loss_fn(outputs, labels)
				validation_loss.append(loss.item())
				prediction = outputs.data.max(1)[1]
				accuracy = prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
				validation_accuracy.append(accuracy.item())
		
		avg_train_loss = sum(training_loss)/len(training_loss)
		avg_valid_loss = sum(validation_loss)/len(validation_loss)

		print('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f|\
			\n| Train Acc:%.2f| Valid Acc:%.2f|time: %.2fs|\n'
			%(epoch, sum(training_loss) / len(training_loss), sum(validation_loss) / len(validation_loss),
			 sum(training_accuracy) / len(training_accuracy), sum(validation_accuracy) / len(validation_accuracy),
			 (end_time - st_time)))
		
		logging.info('\n |Epoch: %d over| Train Loss: %.5f| Valid Loss: %.5f|\
			\n| Train Acc:%.2f| Valid Acc:%.2f| time: %.2fs|\n'
			%(epoch, sum(training_loss) / len(training_loss), sum(validation_loss) / len(validation_loss),
			 sum(training_accuracy) / len(training_accuracy), sum(validation_accuracy) / len(validation_accuracy),
			 (end_time - st_time)))

		state = {
				'epoch':epoch,
				'opt': opt,
				'train_loss': sum(training_loss) / len(training_loss),
				'valid_loss': sum(validation_loss) / len(validation_loss),
				'train_accuracy': sum(training_accuracy) / len(training_accuracy),
				'valid_accuracy': sum(validation_accuracy) / len(validation_accuracy),
				'model_state_dict':model.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'lr':optimizer.param_groups[0]['lr']
				}
		torch.save(state,opt.checkpoints_dir + "net_" + str(epoch) + ".pt")