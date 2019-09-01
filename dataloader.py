
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from imageio import imread 
import torch

import numpy as np
import numpy.random as npr
from PIL import Image
import random
import time
import pdb
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes import NuScenesExplorer 
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import os 

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d import Axes3D

from utils import get_pointcloud 

import cv2 

from logger import Logger 

class nuscenes_dataloader(data.Dataset):
  def __init__(self, batch_size, num_classes, training=True, normalize=None):
    self._num_classes = num_classes
    self.training = training
    self.normalize = normalize
    self.batch_size = batch_size
    self.data_path = "/data/sets/nuscenes"
    self.nusc= NuScenes(version='v1.0-trainval', dataroot = self.data_path, verbose= True)
    self.explorer = NuScenesExplorer(self.nusc)
    self.classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')

    PATH = self.data_path + '/annotations_list.txt'

    with open(PATH) as f:
        self.token = [x.strip() for x in f.readlines()]
    self.token = self.token[:400]

  def __getitem__(self, index):
    # gather tokens and samples needed for data extraction
    tokens = self.token[index]
    im_token = tokens.split('_')[0]
    annotation_token = tokens.split('_')[1]
    
    sample_data = self.nusc.get('sample_data', im_token)
    image_name = sample_data['filename']
    sample = self.nusc.get('sample', sample_data['sample_token'])
    lidar_token = sample['data']['LIDAR_TOP']
    
    # get the sample_data for the image batch
    image_path = '/data/sets/nuscenes/' + image_name
    img = imread('/data/sets/nuscenes/' + image_name)
    im = np.array(img)
    
    # get ground truth boxes 
    _, boxes, camera_intrinsic = self.nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)
    
    for box in boxes:
        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)
        if box.token == annotation_token:
            # Find the crop area of the box 
            width = corners[0].max() - corners[0].min()
            height = corners[1].max() - corners[1].min()
            x_mid = (corners[0].max() + corners[0].min())/2
            y_mid = (corners[1].max() + corners[1].min())/2
            side = max(width, height)*random.uniform(1.0,1.2)
            
            if (x_mid - side/2) < 0:
               side = x_mid*2 
            if (y_mid - side/2) < 0:
               side = y_mid*2
            
            # Crop the image
            bottom_left = [int(x_mid - side/2), int(y_mid - side/2)]
            top_right = [int(x_mid + side/2), int(y_mid + side/2)]
            corners[0]=corners[0] - bottom_left[0]
            corners[1]=corners[1] - bottom_left[1]
            crop_img = im[bottom_left[1]:top_right[1],bottom_left[0]:top_right[0]]
                   
            # Scale to same size             
            scale = 128/ side
            scaled = cv2.resize(crop_img, (128, 128))
            crop_img = np.transpose(scaled, (2,0,1))
            crop_img = crop_img.astype(np.float32)
            crop_img /= 255
            
            # Get corresponding point cloud for the crop
            pcl, m, offset, camera_intrinsic, box_corners = get_pointcloud(self.nusc, bottom_left, top_right, box, lidar_token, im_token)
            break

    pcl = pcl.astype(np.float32)
    box_corners = box_corners.astype(np.float32)
    
    return crop_img, pcl, offset, m, camera_intrinsic, box_corners

  def __len__(self):
    return len(self.token)
