from typing import Tuple
import numpy as np
from matplotlib.axes import Axes
import os.path as osp
from PIL import Image
from pyquaternion import Quaternion
import pdb

import torch
from torch.utils.data.sampler import Sampler
import torch.nn as nn

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

class sampler(Sampler):
   def __init__(self, train_size, batch_size):
      self.num_data = train_size
      self.num_per_batch = int(train_size / batch_size)
      self.batch_size = batch_size
      self.range = torch.arange(0,batch_size).view(1, batch_size).long()
      self.leftover_flag = False
      if train_size % batch_size:
         self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
         self.leftover_flag = True

   def __iter__(self):
      rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
      self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

      self.rand_num_view = self.rand_num.view(-1)
      
      if self.leftover_flag:
         self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

      return iter(self.rand_num_view)

   def __len__(self):
      return self.num_data

def get_pointcloud(nusc, bottom_left, top_right, pointsensor_token: str, camera_token: str, min_dist: float = 1.0) -> Tuple:
   """
   Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
   plane.
   :param pointsensor_token: Lidar/radar sample_data token.
   :param camera_token: Camera sample_data token.
   :param min_dist: Distance from the camera below which points are discarded.
   :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
   """

   cam = nusc.get('sample_data', camera_token)
   pointsensor = nusc.get('sample_data', pointsensor_token)
   pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
   pc = LidarPointCloud.from_file(pcl_path)
   original_pc = np.copy(pc.points)
   im = Image.open(osp.join(nusc.dataroot, cam['filename']))

   # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
   # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
   cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
   pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
   pc.translate(np.array(cs_record['translation']))

   # Second step: transform to the global frame.
   poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
   pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
   pc.translate(np.array(poserecord['translation']))

   # Third step: transform into the ego vehicle frame for the timestamp of the image.
   poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
   pc.translate(-np.array(poserecord['translation']))
   pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

   # Fourth step: transform into the camera.
   cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
   pc.translate(-np.array(cs_record['translation']))
   pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

   # Fifth step: actually take a "picture" of the point cloud.
   depths = pc.points[2, :]

   # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
   points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
   crop_points = pc.points[:3, :]
   # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
   # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
   # casing for non-keyframes which are slightly out of sync.
   mask = np.ones(depths.shape[0], dtype=bool)
   mask = np.logical_and(mask, depths > min_dist)
   mask = np.logical_and(mask, points[0, :] > 1)
   mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
   mask = np.logical_and(mask, points[1, :] > 1)
   mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
   
   points = points[:, mask]
   crop_points = crop_points[:, mask]

   crop_mask = np.ones(crop_points[2,:].shape[0], dtype=bool)
   crop_mask = np.logical_and(crop_mask, points[1,:]>bottom_left[1])
   crop_mask = np.logical_and(crop_mask, points[1,:]<top_right[1])
   crop_mask = np.logical_and(crop_mask, points[0,:]>bottom_left[0])
   crop_mask = np.logical_and(crop_mask, points[0,:]<top_right[0])

   cropped = crop_points[:, crop_mask]

   if np.shape(cropped)[0]>400:
      cropped = cropped.transpose()
      cropped = cropped[np.random.choice(cropped.shape[0],400,replace=False),:]
      cropped = cropped.transpose() 
   return cropped

def render_box(
         corners,
         axis: Axes,
         view: np.ndarray = np.eye(3),
         normalize: bool = False,
         colors: Tuple = ('b', 'r', 'k'),
         linewidth: float = 2):
   """
   Renders the box in the provided Matplotlib axis.
   :param axis: Axis onto which the box should be drawn.
   :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
   :param normalize: Whether to normalize the remaining coordinate.
   :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
      back and sides.
   :param linewidth: Width in pixel of the box sides.
   """

   def draw_rect(selected_corners, color):
      prev = selected_corners[-1]
      for corner in selected_corners:
         axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
         prev = corner

   # Draw the sides
   for i in range(4):
      axis.plot([corners[i][0], corners[i + 4][0]],
              [corners[i][1], corners[i + 4][1]],
              color=colors[2], linewidth=linewidth)

   # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
   draw_rect(corners[:4], colors[0])
   draw_rect(corners[4:], colors[1])

   # Draw line indicating the front
   center_bottom_forward = np.mean(corners[2:4], axis=0)
   center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
   axis.plot([center_bottom[0], center_bottom_forward[0]],
           [center_bottom[1], center_bottom_forward[1]],
           color=colors[0], linewidth=linewidth)
