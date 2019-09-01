import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

from logger import Logger 

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from VPF import PointFusion_Dense_Resnet as PointFusion
from dataloader import nuscenes_dataloader
from utils import ResNet50Bottom, sampler, render_box, render_pcl, visualize_result, IoU

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os 

logger = Logger('./logs/Dense_Resnet')

nusc_classes = ['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']

batch_size = 4             
nusc_set = nuscenes_dataloader(batch_size, len(nusc_classes), training = True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = batch_size, shuffle=True)
nusc_iters_per_epoch = int(len(nusc_set) / batch_size)

num_epochs = 50

model = PointFusion(k = 1, feature_transform = False)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma=0.5)

regressor = nn.SmoothL1Loss(reduction='none')

im = torch.FloatTensor(1).cuda()
points = torch.FloatTensor(1).cuda()
offset = torch.FloatTensor(1).cuda()
m = torch.FloatTensor(1).cuda()
rot_matrix = torch.FloatTensor(1).cuda()
gt_corners = torch.FloatTensor(1).cuda()

im = Variable(im)
points = Variable(points)
offset = Variable(offset)
m= Variable(m)
rot_matrix = Variable(rot_matrix)
gt_corners = Variable(gt_corners)

date = '08_28_2019__2'

out_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = out_dir + '/trained_model/' + date
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

for epoch in range(1, num_epochs+1):
   scheduler.step()
   nusc_iter = iter(nusc_dataloader)
   loss_temp = 0
   loss_epoch = 0
   for step in range(nusc_iters_per_epoch):
      data = next(nusc_iter)
      with torch.no_grad():
          im.resize_(data[0].size()).copy_(data[0])
          points.resize_(data[1].size()).copy_(data[1])
          offset.resize_(data[2].size()).copy_(data[2])
          m.resize_(data[3].size()).copy_(data[3])
          rot_matrix.resize_(data[4].size()).copy_(data[4])
          gt_corners.resize_(data[5].size()).copy_(data[5]) 
      
      optimizer.zero_grad()
      model = model.train()
      pred_offset, scores = model(im, points)

      loss = 0
      n = 400

      # Unsupervised loss 
      loss = regressor(pred_offset, offset).mean(dim=(2,3))*scores -0.1*torch.log(scores)    
      loss = loss.sum(dim=1)/n 
      loss = loss.sum(dim=0)/batch_size

      loss_temp += loss.item()
      loss_epoch += loss.item()

      loss.backward()
      optimizer.step()  
      
      # Finding anchor point and predicted offset based on maximum score 
      max_inds = scores.max(dim=1)[1].cpu().numpy()
      p_offset = np.zeros((4,8,3))
      anchor_points = np.zeros((4,3))
      truth_boxes = np.zeros((4,8,3))
      for i in range(0,4):
         p_offset[i] = pred_offset[i][max_inds[i]].cpu().detach().numpy() 
         anchor_points[i] = (points.cpu().numpy().transpose((0,2,1)))[i][max_inds[i]]
         truth_boxes[i] = gt_corners[i].cpu().numpy()

      # visualize_result(p_offset, anchor_points, truth_boxes)
      if step%10 == 0 and step!=0:
         loss_temp/= 10
         print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
               .format(epoch, num_epochs+1, step, nusc_iters_per_epoch, loss_temp))
         loss_temp = 0
   loss_epoch /= nusc_iters_per_epoch
   logger.scalar_summary('loss', loss_epoch, epoch)

   print("Loss for Epoch {} is {}".format(epoch, loss_epoch))
   loss_epoch = 0
