
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
data_path = "/data/sets/nuscenes"

nusc= NuScenes(version='v1.0-trainval', dataroot = data_path, verbose= True)
explorer = NuScenesExplorer(nusc)

PATH = data_path + '/CAMFRONT.txt'

with open(PATH) as f:
    image_token = [x.strip() for x in f.readlines()]

image_token = image_token[:1000]
annotations = []
counter = 0
#pdb.set_trace()
for im_token in image_token:
    print(counter)
    sample_data = nusc.get('sample_data', im_token)
    sample = nusc.get('sample', sample_data['sample_token'])
    lidar_token = sample['data']['LIDAR_TOP']
    
    # get ground truth boxes 
    _, boxes, camera_intrinsic = nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)

    for box in boxes:
        visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
        vis_level = int(nusc.get('visibility', visibility_token)['token'])
        if (vis_level == 3) or (vis_level == 4):
            #print(vis_level)
            visible = True
        else:
            visible = False 
        corners = view_points(box.corners(), view=camera_intrinsic, normalize=True)
        if (visible == True) and ((corners[0].max() - corners[0].min())> 64) and ((corners[1].max() - corners[1].min())> 64):
            bottom_left = [np.int(corners[0].min()), np.int(corners[1].min())]
            top_right = [np.int(corners[0].max()), np.int(corners[1].max())]
            if box.name.split('.')[0] == 'vehicle':
                if box.name.split('.')[1] != 'emergency':
                    name = box.name.split('.')[1]
                else:
                    name = ''
            elif box.name.split('.')[0] == 'human':
                name = 'pedestrian'
            elif box.name.split('.')[0] == 'movable_object':
                if box.name.split('.')[1] != 'debris' and box.name.split('.')[1] != 'pushable_pullable': 
                    name = box.name.split('.')[1]
                else:
                    name = ''
            else:
                name = ''
            pcl, _, _, _, _ = get_pointcloud(nusc, bottom_left, top_right, box, lidar_token, im_token)
            #print(np.shape(pcl)[1])
            if (name == 'car' or name == 'pedestrian') and len(pcl)!=0:
                if np.shape(pcl)[1] == 400: 
                    annotation_token = box.token
                    annotations = annotations + [im_token + "_"+ annotation_token]
    counter = counter + 1

print("Saving...")
with open('/data/sets/nuscenes/car_pedestrian_annotations_list.txt', 'w') as f:
    for item in annotations:
        f.write("%s\n" % item)
