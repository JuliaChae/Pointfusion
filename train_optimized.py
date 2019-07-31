import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from dataloader import nuscenes_dataloader
from utils import sampler, render_box

import matplotlib.pyplot as plt
import numpy as np
import pdb


def train(rank, res50_conv2, pointfeat, trans, model, epochs):
   torch.manual_seed(1 + rank)
   nusc_classes = ['__background__', 
                              'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
   nusc_sampler_batch = sampler(1939, 1)
   nusc_set = nuscenes_dataloader(1, len(nusc_classes), training = True)
   nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = 1, sampler = nusc_sampler_batch, num_workers = 0, drop_last = True)
   nusc_iters_per_epoch = int(len(nusc_set) / 1)

   optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
   criterion = nn.SmoothL1Loss()

   for epoch in range(1, epochs + 1):
      train_epoch(epoch, model, res50_conv2, pointfeat, trans, nusc_dataloader, optimizer, criterion)

def train_epoch(epoch, model, res50_conv2, pointfeat, trans, data_loader, optimizer, criterion)
   im = torch.FloatTensor(1)
   corners = torch.FloatTensor(1)
   points = torch.FloatTensor(1)

   im = im.cuda()
   corners = corners.cuda()
   points = points.cuda()

   im = Variable(im)
   corners = Variable(corners)
   points = Variable(points)

   nusc_iter = iter(nusc_dataloader)
   model.train()
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

            optimizer.zero_grad()
            out = trans(points)
            
            loss_stn = feature_transform_regularizer(out)
            loss_smoothL1 = criterion(pred_box, corners)
            loss = loss_stn + loss_smoothL1

            loss.backward()
            optimizer.step()

      if step%100 == 0:
         print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
               .format(epoch+1, num_epochs, step+1, nusc_iters_per_epoch, loss.item()))