
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

    if training == True: 
        PATH = self.data_path + '/train_mini_diffcam.txt'
    else:
        PATH = self.data_path + '/test_mini_diffcam.txt'

    with open(PATH) as f:
        self.image_token = [x.strip() for x in f.readlines()]

  def __getitem__(self, index):

    # gather tokens and samples needed for data extraction
    im_token = self.image_token[index]
    sample_data = self.nusc.get('sample_data', im_token)
    image_name = sample_data['filename']
    sample = self.nusc.get('sample', sample_data['sample_token'])
    lidar_token = sample['data']['LIDAR_TOP']

    # get the sample_data for the image batch
    image_path = '/data/sets/nuscenes/' + image_name
    img = imread('/data/sets/nuscenes/' + image_name)
    im = np.array(img)
    
    # get ground truth boxes 
    _, boxes, camera_intrinsic = self.nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ANY)
    gt_corners = list(box.corners() for box in boxes)

    # iterate through boxes and store roi crops, 3d bounding boxes and point clounds that appear in the image.
    crops = []
    box_corners = [] 
    pointclouds = [] 
    #print(len(boxes))
    for box in boxes:
        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)
        if (corners[0].min() > 0.0) and (corners[1].min()>0.0) and (corners[0].max()>0.0) and (corners[1].max() > 0.0):
            bottom_left = [np.int(corners[0].min()), np.int(corners[1].min())]
            top_right = [np.int(corners[0].max()), np.int(corners[1].max())]
            corners[0]=corners[0] - bottom_left[0]
            corners[1]=corners[1] - bottom_left[1]
            box_corners = box_corners + [corners.astype(int).transpose()]
            crop_img = im[bottom_left[1]:top_right[1],bottom_left[0]:top_right[0]]
            crop_img = np.transpose(crop_img, (2,0,1))
            pcl = get_pointcloud(self.nusc, bottom_left, top_right, lidar_token, im_token)
            pointclouds = pointclouds + [pcl]
            crops = crops + [crop_img.astype(int)]

    return crops, box_corners, pointclouds

  def draw_rect(self,axis, selected_corners):
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]])
        prev = corner
    # Function checks if boxes appear in image and converts it to image frame 

  def display(self,pc):
    # 3D plotting of point cloud 
    fig=plt.figure()
    ax = fig.gca(projection='3d')

    #ax.set_aspect('equal')
    X = pc[0]
    Y = pc[1]
    Z = pc[2]
    c = pc[3]

    """
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    X = X[np.where(radius<20)]
    Y = Y[np.where(radius<20)]
    Z = Z[np.where(radius<20)]
    c = pc.points[3][np.where(radius<20)]
    print(radius)
    """
    ax.scatter(X, Y, Z, s=1, c=cm.hot((c/100)))

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    i=0
    for xb, yb, zb in zip(Xb, Yb, Zb):
        i = i+1 
        ax.plot([xb], [yb], [zb], 'w')

  def __len__(self):
    return len(self.image_token)
