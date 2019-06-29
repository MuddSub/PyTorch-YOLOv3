from __future__ import division
import sklearn.metrics as skm
from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable
import numpy as np
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

def test(opt):
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
	'''
	print(len(dataloader))
	# Initiate YOLO model
	yolo = Darknet(opt.model_def).to(device)
	yolo.load_state_dict(torch.load("checkpoints/test.pth"))
	yolo.eval()
	model = yolo_process(yolo)
	'''
	print(len(dataloader))
	

	timeList=[]	
	pred = []
	predGate=[]
	count_none = 0
	error = 50
	for batch_i, (path, input,label) in enumerate(dataloader):
		
		# expect x1, x2 in order
		print("path  "+str(path[0]))
		#output1, output2 = model(input)
		img = cv2.imread("/Users/rongk/Downloads/visionCode/PyTorch-YOLOv3/"+path[0])
		img = cv2.resize(img,(640,640))
		#cv2.imshow("img",img)
		start=time.time()
		output = mainImg(img)
		timeCost = time.time()-start
		print("time cost "+ str(timeCost))

		timeList.append(timeCost)
		#print("debug output")
		print(output)
		result1=0
		result2=0
		_,_,label1,label2,_,_=label[0]
		label1,label2 = abs(label1.item()),abs(label2.item())
		
		if len(output) == 0:
			if label1<=640:
				pred.append(0)
			if label2<=640:
				pred.append(0)
			count_none+=1
			output1 = -1
			output2 = -1
		elif len(output) == 1:
			output1 = output[0]
			output2 = -1
			result1=1 if (abs(label1-output1)<error or abs(label1 - output2)<error) else 0	
			pred.append(result1)
			if label1<=640 and label2<=640:
				pred.append(0)
		elif len(output)==2 and not None in output:
			output1,output2 = output
			if output1>output2:
				output1,output2 = output2, output1	
				
			result1 = 1 if (abs(label1 - output1 ) < error or abs(label2-output1)<error ) else 0
			

			result2 = 1 if (abs(label2 - output2 ) <error or abs(label2- output1)<error) else 0
			
			

			pred.append(result1)
			pred.append(result2)
		print("label!")
		print((label1,label2))
		print("outputs!")
		print((output1,output2))
		print(result1)
		print(result2)
		predGate.append( result1 or result2)
	print(pred)
	ideal = [1]*len(pred)
	print(ideal)
	print(predGate)
	idealGate=[1]*len(predGate)
	f1 = skm.f1_score(ideal,pred, average = "binary")
	f1g = skm.f1_score(idealGate,predGate,average="binary")
	print("f1 one side :   "+str(f1))
	print("f1 gate : "+str(f1g))
	print("count none "+str(count_none))
	print("avg time :"+str(np.array(timeList).mean()))
	return [f1,f1g,np.array(timeList).mean()]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
	parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
	parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
	parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
	parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
	parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
	parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
	parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
	parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
	parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
	parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
	opt = parser.parse_args()
	epoch = 10
	result=np.array([0.0,0.0,0.0])
	print(opt)
	for i in range(epoch):
		result+=np.array(test(opt))
	print(np.array(result)/epoch)
	
