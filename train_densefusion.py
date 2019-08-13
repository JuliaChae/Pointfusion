import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

from Pointnet import PointNetfeat, STN3d, feature_transform_regularizer
from MLP import MLP_Dense as MLP
from dataloader import nuscenes_dataloader
from utils import ResNet50Bottom, sampler, render_box

import matplotlib.pyplot as plt
import numpy as np
import pdb

import os 

nusc_classes = ['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck']
batch_size = 1             
nusc_sampler_batch = sampler(200, 1)
nusc_set = nuscenes_dataloader(1, len(nusc_classes), training = True)
nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = 1, sampler = nusc_sampler_batch, num_workers = 0)
nusc_iters_per_epoch = int(len(nusc_set) / 1)

num_epochs = 200

res50_model = models.resnet50(pretrained=True)
res50_model.cuda()
res50_model.eval()
res50_conv2 = ResNet50Bottom(res50_model)

globalfeat = PointNetfeat(global_feat=True)
pointfeat = PointNetfeat(global_feat=False)
pointfeat.cuda()
globalfeat.cuda()

model = MLP()
model.cuda()

trans = STN3d()
trans.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.SmoothL1Loss()

im = torch.FloatTensor(1)
corners = torch.FloatTensor(1)
points = torch.FloatTensor(1)
classes = torch.LongTensor(1)
fusion_feat = torch.FloatTensor(2)

im = im.cuda()
corners = corners.cuda()
points = points.cuda()
classes = classes.cuda()
fusion_feat = fusion_feat.cuda()

im = Variable(im)
corners = Variable(corners)
points = Variable(points)
classes = Variable(classes)

date = '08_12_2019__4'

out_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = out_dir + '/trained_model/' + date
if not os.path.exists(output_dir):
      os.makedirs(output_dir)

"""
f1 = open(output_dir + '/loss.txt','w+')
f1.write("epoch    train\n")
f1.close()
 """  
min_loss = 100
for epoch in range(1, num_epochs+1):
   nusc_iter = iter(nusc_dataloader)
   model.train()
   loss_temp = 0
   loss_epoch = 0
   for step in range(nusc_iters_per_epoch):
      data = next(nusc_iter)
      #pdb.set_trace()
      with torch.no_grad():
          im.resize_(data[0].size()).copy_(data[0])
          corners.resize_(data[1].size()).copy_(data[1])
          points.resize_(data[2].size()).copy_(data[2])
          classes.resize_(data[3].size()).copy_(data[3])
    
      base_feat = res50_conv2(im)
      base_feat = torch.squeeze(base_feat,2)
      base_feat = torch.squeeze(base_feat,2)

      #print(points.size())
      global_feat, _ = globalfeat(points)
      point_feat, _ = pointfeat(points)
      
      for point in point_feat: 
        #torch.cat([point, global_feat, base_feat], dim=1)
        fusion_point = torch.cat([point.unsqueeze(0), global_feat, base_feat], dim=1)
        if fusion_feat.size() == torch.Size([2]):
            fusion_feat = fusion_point
        else:
            fusion_feat = torch.cat([fusion_feat, fusion_point], dim=0)

      offset, score = model(fusion_feat)
      
      #bbox_loss = criterion(pred_box, corners)
      #cls_loss = F.cross_entropy(pred_class, classes)
      #loss = bbox_loss.mean() + cls_loss.mean()

      #loss_temp += loss.item()
      #loss_epoch += loss.item()
       
      #pdb.set_trace()
      #optimizer.zero_grad()
      #loss.backward()
      #optimizer.step()

   """"
      if step%100 == 0 and step!=0:
         loss_temp /= 100 
         print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
               .format(epoch, num_epochs+1, step, nusc_iters_per_epoch, loss_temp))
         loss_temp = 0
   loss_epoch /= nusc_iters_per_epoch
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
   f1 = open(output_dir + '/loss.txt','a+')
   f1.write("%2d        %.4f\n" % (epoch, loss_epoch))
   f1.close()
   loss_epoch = 0
   """
