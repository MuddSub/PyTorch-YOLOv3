from __future__ import division
import sklearn.metrics as skm
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from gate import *

def test(opt,dataloader):

	# Initiate YOLO model
	model = Darknet(opt.model_def).to(device)
	model.load_state_dict(torch.load("checkpoints/test.pth"))
	model.eval()

	
	timeList=[]
	pred = []
	predGate=[]
	predGate2=[]
	count_none = 0
	error = 50
	for batch_i, (path, input,label) in enumerate(dataloader):
		
		# expect x1, x2 in order
		print("path  "+str(path[0]))
		#output1, output2 = model(input)
		#img = cv2.imread("/Users/rongk/Downloads/visionCode/PyTorch-YOLOv3/"+path[0])
		#img = cv2.resize(img,(640,640))
		start = time.time()
		output = model(input)
		output = non_max_suppression(output, 0.2,0.5)
		timeCost = time.time()-start
		timeList.append(timeCost)
		print(output)
		result1=0
		result2=0
		_,_,label1,label2,_,_=label[0]
		label1,label2 = abs(label1.item()),abs(label2.item())
		
		if not type(output[0]) == type(torch.tensor([1])):
			if label1<=640:
				pred.append(0)
			if label2<=640:
				pred.append(0)
			count_none+=1
			output1 = -1
			output2 = -1
		else:
			output = output[0].numpy().tolist()
			output0 =[]
			for i in output:
				output0.append(i)
			output = output0
			lowest_value = 0
			for i in output:
				if lowest_value<i[4]:
					lowest_value = i[4]

				output1= i[0]
				output2 = i[2]
			if output1>output2:
				output1,output2 = output2, output1	
				
			result1 = 1 if (abs(label1 - output1 ) < error or abs(label2-output1)<error ) else 0
			

			result2 = 1 if (abs(label2 - output2 ) <error or abs(label1- output2)<error) else 0
			
			

			pred.append(result1)
			pred.append(result2)
		print("label!")
		print((label1,label2))
		print("outputs!")
		print((output1,output2))
		print(result1)
		print(result2)
		predGate.append(result1 or result2)	
		predGate2.append(result1 and result2)
	print(pred)
	ideal = [1]*len(pred)
	print(ideal)	
	idealGate = [1] * len(predGate)

	f1 = skm.f1_score(ideal,pred, average = "binary")
	f1g = skm.f1_score(idealGate,predGate,average="binary")
	f1g2 = skm.f1_score([1]*len(predGate2),predGate2, average="binary")
	print("f1  :   "+str(f1))
	print("f1 one side: "+str(f1g))
	print("f1 both sides "+str(f1g2))
	print("count none "+str(count_none))
	print("time :"+str(np.array(timeList).mean()))
	return [f1,f1g,f1g2,np.array(timeList).mean()]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
	parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
	parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
	parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
	parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
	parser.add_argument("--pretrained_weights", type=str, default="checkpoints/test.pth", help="if specified starts from checkpoint model")
	parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
	parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
	parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
	parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
	parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
	opt = parser.parse_args()	
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Get data configuration
	data_config = parse_data_config(opt.data_config)
	train_path = data_config["train"]
	valid_path = data_config["valid"]
	class_names = load_classes(data_config["names"])

	
	# If specified we start from checkpoint
	if False: #opt.pretrained_weights:
		if opt.pretrained_weights.endswith(".pth"):
			model.load_state_dict(torch.load(opt.pretrained_weights))
		else:
			model.load_darknet_weights(opt.pretrained_weights)
	
	# Get dataloader
	dataset = ListDataset("data/custom/test.txt", augment=True, multiscale=opt.multiscale_training)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.n_cpu,
		pin_memory=True,
		collate_fn=dataset.collate_fn
	)
	print(len(dataloader))
	
	
	
	epoch=1
	#result=np.array([0.0,0.0,0.0,0.0])
	f1=[] # number of bars detected
	f1g=[] #at least one bar detected per gate
	f1g2=[] #both bars detected per gate
	t=[] #time
	print(opt)
	for i in range(epoch):
		a,b,c,d =test(opt,dataloader)
		f1.append(a)
		f1g.append(b)
		f1g2.append(c)
		t.append(d)
	print("f1,f1g,f1g2,t. mean and std")
	print("f1")
	f1=np.array(f1)
	f1g=np.array(f1g)
	f1g2=np.array(f1g2)
	t=np.array(t)
	print(f1.mean())
	print(f1.std())
	print("f1g")
	print(f1g.mean())
	print(f1g.std())
	print("f1g2")
	print(f1g2.mean())
	print(f1g2.std())
	print("t")
	print(t.mean())
	print(t.std())
