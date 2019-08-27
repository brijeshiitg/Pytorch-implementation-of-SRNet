import torch
import numpy as np
import torch.nn as nn
from scipy import misc
from glob import glob
from model import Srnet

TEST_BATCH_SIZE = 40
COVER_PATH = '/path/to/cover/images/'
STEGO_PATH = '/path/to/stego/images/'
CHKPT='SRNet_model_weights.pt'

cover_image_names = glob(COVER_PATH)
stego_image_names = glob(STEGO_PATH)

cover_labels = np.zeros((len(cover_image_names)))
stego_labels = np.ones((len(stego_image_names)))

model = Srnet().cuda()

ckpt = torch.load(CHKPT)
model.load_state_dict(ckpt['model_state_dict'])

images = torch.empty((TEST_BATCH_SIZE,1,256,256), dtype=torch.float)
test_accuracy = []

for idx in range(0,len(cover_image_names), TEST_BATCH_SIZE//2):
	cover_batch = cover_image_names[idx:idx+TEST_BATCH_SIZE//2]
	stego_batch = stego_image_names[idx:idx+TEST_BATCH_SIZE//2]

	batch = []
	batch_labels = []

	x_i=0
	y_i=0
	for i in range(2*len(cover_batch)):
		if i%2==0:
			batch.append(stego_batch[x_i])
			batch_labels.append(1)
			x_i+=1
		else:
			batch.append(cover_batch[y_i])
			batch_labels.append(0)
			y_i+=1

	for i in range(TEST_BATCH_SIZE):
		images[i,0,:,:]= torch.tensor(misc.imread(batch[i])).cuda()
	image_tensor = images.cuda()
	batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()

	outputs = model(image_tensor)
	prediction = outputs.data.max(1)[1]

	accuracy = prediction.eq(batch_labels.data).sum()*100.0/(batch_labels.size()[0])
	test_accuracy.append(accuracy.item())

print("test_accuracy = %.2f"%(sum(test_accuracy)/len(test_accuracy)))