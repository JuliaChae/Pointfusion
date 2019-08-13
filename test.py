import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP 
from dataloader import nuscenes_dataloader
from utils import ResNet50Bottom, sampler, render_box

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pdb

import os 

def draw_box(axis, corners, colors: Tuple = ('b', 'r', 'k')):
   for i in range(4):
      axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=colors[2], linewidth=2)

   # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
   draw_rect(axis, corners.T[:4], colors[0])
   draw_rect(axis, corners.T[4:], colors[1])

def draw_rect(axis, selected_corners, color):
    prev = selected_corners[-1]
    for corner in selected_corners:
       axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=2)
       prev = corner
       
nusc_classes = ['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
nusc_sampler_batch = sampler(200, 1)
nusc_set = nuscenes_dataloader(1, len(nusc_classes), training = False)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = 1, sampler = nusc_sampler_batch, num_workers = 0)
nusc_iters_per_epoch = int(len(nusc_set) / 1)

epochs = 1

res50_model = models.resnet50(pretrained=True)
res50_model.cuda()
res50_model.eval()
res50_conv2 = ResNet50Bottom(res50_model)

pointfeat = PointNetfeat(global_feat=True)
pointfeat.cuda()

trans = STN3d()
trans.cuda()

"""
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.SmoothL1Loss()
"""

im = torch.FloatTensor(1)
corners = torch.FloatTensor(1)
points = torch.FloatTensor(1)
cls = torch.LongTensor(1)
im_dis = torch.FloatTensor(1)

im = im.cuda()
corners = corners.cuda()
points = points.cuda()
cls = cls.cuda()
im_dis = im_dis.cuda()

im = Variable(im)
corners = Variable(corners)
points = Variable(points)
cls=Variable(cls)
im_dis = Variable(im_dis)

date = '08_12_2019__4'
epoch = 486
in_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = in_dir + '/trained_model/' + date
if not os.path.exists(input_dir):
      raise Exception('There is no input directory for loading network from ')
load_name = os.path.join(input_dir, 'pointfusion_{}_{}.pth'.format(epoch, date))

model = MLP()
print("load checkpoint %s" % (load_name))
checkpoint = torch.load(load_name)
model.load_state_dict(checkpoint['model'])
model.cuda()

with torch.no_grad():
    for epoch in range(0, 50):
       nusc_iter = iter(nusc_dataloader)
       model.eval()
       loss_temp = 0
       for step in range(nusc_iters_per_epoch):
          data = next(nusc_iter)
          im.resize_(data[0].size()).copy_(data[0])
          corners.resize_(data[1].size()).copy_(data[1])
          points.resize_(data[2].size()).copy_(data[2])
          cls.resize_(data[3].size()).copy_(data[3])
          im_dis.resize_(data[4].size()).copy_(data[4])

          image = im_dis[0].cpu().numpy().astype("int32")
          _, ax = plt.subplots(1, 1, figsize=(5, 5))
          ax.imshow(image)
          
          #print(image)
          base_feat = res50_conv2(im)
          base_feat = torch.squeeze(base_feat,2)
          base_feat = torch.squeeze(base_feat,2)
          
          global_feat, _ = pointfeat(points)
          fusion_feat = torch.cat([global_feat, base_feat], dim=1)

          #print(fusion_feat)
          pred_box, pred_class = model(fusion_feat)
          print(pred_box)
          print(pred_class)
          box = pred_box[0].cpu().numpy().T
          #pdb.set_trace()
          draw_box(ax, box)
          plt.show()
      
      
