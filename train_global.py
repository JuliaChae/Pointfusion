import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

from logger import Logger 

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP as MLP
from dataloader import nuscenes_dataloader
from utils import ResNet50Bottom, sampler, render_box, render_pcl, visualize_result, IoU

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os 

logger = Logger('./logs/4')

nusc_classes = ['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
batch_size = 4             
nusc_sampler_batch = sampler(400, 2)
nusc_set = nuscenes_dataloader(batch_size, len(nusc_classes), training = True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = batch_size, sampler = nusc_sampler_batch)
nusc_iters_per_epoch = int(len(nusc_set) / batch_size)

num_epochs = 200

model = MLP()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
regressor = nn.SmoothL1Loss()
classifier = nn.BCELoss()

im = torch.FloatTensor(1)
points = torch.FloatTensor(1)
offset = torch.FloatTensor(1)
m = torch.FloatTensor(1)
rot_matrix = torch.FloatTensor(1)
gt_corners = torch.FloatTensor(1)

im = im.cuda()
points = points.cuda()
offset = offset.cuda()
m = m.cuda()
rot_matrix = rot_matrix.cuda()
gt_corners = gt_corners.cuda()

im = Variable(im)
points = Variable(points)
offset = Variable(offset)
m= Variable(m)
rot_matrix = Variable(rot_matrix)
gt_corners = Variable(gt_corners)


date = '08_26_2019__1'

out_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = out_dir + '/trained_model/' + date
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

min_loss = 100

for epoch in range(1, num_epochs+1):
   nusc_iter = iter(nusc_dataloader)
   loss_temp = 0
   loss_epoch = 0
   model = model.train()

   for step in range(nusc_iters_per_epoch):
      data = next(nusc_iter)
      with torch.no_grad():
          im.resize_(data[0].size()).copy_(data[0])
          points.resize_(data[1].size()).copy_(data[1])
          offset.resize_(data[2].size()).copy_(data[2])
          m.resize_(data[3].size()).copy_(data[3])
          rot_matrix.resize_(data[4].size()).copy_(data[4])
          gt_corners.resize_(data[5].size()).copy_(data[5]) 

      boxes, classes = model(im, points)
      loss = 0
      n = 400
      
      loss = regressor(boxes, gt_corners)  

      loss_temp += loss.item()
      loss_epoch += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()  

      if step%10 == 0 and step!=0:
         loss_temp /= 10
         print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
               .format(epoch, num_epochs+1, step, nusc_iters_per_epoch, loss_temp))
         loss_temp = 0
   loss_epoch /= nusc_iters_per_epoch
   logger.scalar_summary('loss', loss_epoch, epoch)
   
   if loss_epoch < min_loss: 
      min_loss = loss_epoch
      print("Saving model...")
      save_name = os.path.join(output_dir, 'pointfusion_{}_{}.pth'.format(epoch, date))
      torch.save({          'session': date,
              'epoch': epoch + 1,
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict()
               }, save_name)
    
   print("Loss for Epoch {} is {}".format(epoch, loss_epoch))
   loss_epoch = 0
