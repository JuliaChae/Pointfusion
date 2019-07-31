import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP 
from dataloader import nuscenes_dataloader
from utils import ResNet50Bottom, sampler, render_box

import matplotlib.pyplot as plt
import numpy as np
import pdb


nusc_classes = ['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
nusc_sampler_batch = sampler(1939, 1)
nusc_set = nuscenes_dataloader(1, len(nusc_classes), training = True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = 1, sampler = nusc_sampler_batch, num_workers = 0)
nusc_iters_per_epoch = int(len(nusc_set) / 1)

epochs = 1

res50_model = models.resnet50(pretrained=True)
res50_model.cuda()
res50_model.eval()
res50_conv2 = ResNet50Bottom(res50_model)

pointfeat = PointNetfeat(global_feat=True)
pointfeat.cuda()

model = MLP()
model.cuda()

trans = STN3d()
trans.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.SmoothL1Loss()

im = torch.FloatTensor(1)
corners = torch.FloatTensor(1)
points = torch.FloatTensor(1)

im = im.cuda()
corners = corners.cuda()
points = points.cuda()

im = Variable(im)
corners = Variable(corners)
points = Variable(points)

for epoch in range(0, 5):
   nusc_iter = iter(nusc_dataloader)
   model.train()
   loss_temp = 0
   for step in range(nusc_iters_per_epoch):
      crops, boxes, pcl = next(nusc_iter)
      for i in range(len(crops)):
         if pcl[i].nelement()!=0: 
            with torch.no_grad():
               im.resize_(crops[i].size()).copy_(crops[i])
               corners.resize_(boxes[i].size()).copy_(boxes[i])
               points.resize_(pcl[i].size()).copy_(pcl[i])

            base_feat = res50_conv2(im)
            base_feat = torch.squeeze(base_feat,2)
            base_feat = torch.squeeze(base_feat,2)

            global_feat, _ = pointfeat(points)
            fusion_feat = torch.cat([global_feat, base_feat], dim=1)

            pred_box, pred_class = model(fusion_feat)

            out = trans(points)
            loss_stn = feature_transform_regularizer(out)
            loss_smoothL1 = criterion(pred_box, corners)
            loss = loss_stn + loss_smoothL1
            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

      if step%100 == 0:
         loss_temp /= 100 
         print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
               .format(epoch+1, epochs, step+1, nusc_iters_per_epoch, loss_temp))