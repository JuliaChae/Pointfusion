from typing import Tuple
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
import os.path as osp
from PIL import Image
from pyquaternion import Quaternion
import pdb
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data.sampler import Sampler
import torch.nn as nn

from nuscenes import NuScenesExplorer 
from nuscenes.utils.data_classes import LidarPointCloud, Box
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

def get_pointcloud(nusc, bottom_left, top_right, box, pointsensor_token: str, camera_token: str, min_dist: float = 1.0) -> Tuple:
   """
   Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
   plane.
   :param pointsensor_token: Lidar/radar sample_data token.
   :param camera_token: Camera sample_data token.
   :param min_dist: Distance from the camera below which points are discarded.
   :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
   """
   sample_data = nusc.get("sample_data", camera_token)
   explorer = NuScenesExplorer(nusc)
   
   cam = nusc.get('sample_data', camera_token)
   pointsensor = nusc.get('sample_data', pointsensor_token)
   
   im = Image.open(osp.join(nusc.dataroot, cam['filename']))
   
   sample_rec = explorer.nusc.get('sample', pointsensor['sample_token'])
   chan = pointsensor['channel']
   ref_chan = 'LIDAR_TOP'
   pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan, nsweeps = 10)
   data_path, boxes, camera_intrinsic = nusc.get_sample_data(pointsensor_token, selected_anntokens=[box.token])
   pcl_box = boxes[0]
   
   original_points = pc.points.copy()
   
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
   center = np.array([[box.center[0]],[box.center[1]],[box.center[2]]])
   box_center = view_points(center, np.array(cs_record['camera_intrinsic']), normalize=True)
   box_corners = box.corners()
   
   # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
   # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
   # casing for non-keyframes which are slightly out of sync.
   mask = np.ones(depths.shape[0], dtype=bool)
   mask = np.logical_and(mask, depths > min_dist)
   mask = np.logical_and(mask, points[0, :] > 1)
   mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
   mask = np.logical_and(mask, points[1, :] > 1)
   mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
   
   original_points = original_points[:, mask]
   points = points[:, mask]
   points_2 = pc.points[:, mask] 
   
   crop_mask = np.ones(points.shape[1], dtype=bool)
   crop_mask = np.logical_and(crop_mask, points[1,:]>bottom_left[1])
   crop_mask = np.logical_and(crop_mask, points[1,:]<top_right[1])
   crop_mask = np.logical_and(crop_mask, points[0,:]>bottom_left[0])
   crop_mask = np.logical_and(crop_mask, points[0,:]<top_right[0])
   
   original_points = original_points[:, crop_mask]
   points = points[:, crop_mask]
   points_3 = points_2[:, crop_mask]
   
   image_center = np.asarray([((bottom_left[0] + top_right[0])/2),((bottom_left[1] + top_right[1])/2), 1])
   
   # rotate to make the ray passing through the image_center into the z axis 
   z_axis = np.linalg.lstsq(np.array(cs_record['camera_intrinsic']), image_center, rcond = None)[0]
   v = np.asarray([z_axis[0], z_axis[1], z_axis[2]])
   z = np.asarray([0., 0., 1.])
   normal = np.cross(v, z)
   theta = np.arccos(np.dot(v, z)/np.sqrt(v.dot(v)))
   new_pts = [] 
   new_corner = []
   old_pts = []
   points_3 = points_3[:3, :]
   translate = np.dot(rotation_matrix(normal, theta), image_center)
   for point in points_3.T: 
       new_pts = new_pts + [np.dot(rotation_matrix(normal, theta), point)]
   for corner in box_corners.T:
       new_corner = new_corner + [np.dot(rotation_matrix(normal, theta), corner)]
   
   points = np.asarray(new_pts)
   original_points = original_points[:3, :].T
   new_corners = np.asarray(new_corner)

   reverse_matrix = rotation_matrix(normal, -theta)

   # Sample 400 points
   if np.shape(new_pts)[0]>400:
      mask = np.random.choice(points.shape[0], 400, replace=False)
      points = points[mask, :]
      original_points= original_points[mask, :]

   shift = np.expand_dims(np.mean(points, axis = 0), 0)
   points = points - shift # center
   new_corners = new_corners - shift # center
   dist = np.max(np.sqrt(np.sum(points ** 2, axis = 1)),0)
   points = points / dist #scale
   new_corners = new_corners / dist #scale
    
   # Compute offset from point to corner 
   n= np.shape(points)[0]
   offset= np.zeros((n, 8, 3))
   for i in range(0, n):
      for j in range(0, 8):
         offset[i][j] = new_corners[j] - points[i]
   
   # Compute mask on which points lie inside of the box 
   m = []        
   for point in original_points:
       if in_box(point, pcl_box.corners()) == True:
            m = m + [1]
       else:
            m = m + [0]
   m = np.asarray(m)

   return points.T, m, offset, reverse_matrix, new_corners 
   
def in_box(point, box):
    x_max = box[0].max()
    x_min = box[0].min()
    y_max = box[1].max()
    y_min = box[1].min()
    z_max = box[2].max()
    z_min = box[2].min()
    if point[0]<x_max and point[0]>x_min and point[1]<y_max and point[1]>y_min and point[2]<z_max and point[2]>z_min:
        return True
    else:
        return False 
         
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
                     
def render_pcl(pc, box =  [], name = "default"):
   print(pc)
   fig=plt.figure(name)
   ax = fig.gca(projection='3d')
    
   X = pc[0]
   Y = pc[1]
   Z = pc[2]
   ax.scatter(X, Y, Z, s=1)
   
   # pcl characteristics 
   print("X maximum is :" + str(X.max()))
   print("Y maximum is :" + str(Y.max()))
   print("Z maximum is :" + str(Z.max()))

   max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
   Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
   Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
   Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

   i=0
   for xb, yb, zb in zip(Xb, Yb, Zb):
      i = i+1
      ax.plot([xb], [yb], [zb], 'b')
   if len(box)!=0:
       x = box[0] 
       y = box[1]
       z = box[2]
       ax.plot(x, y, z, color = 'r')
   return ax 
  
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

def visualize_result(anchor_point, offset, gt_boxes):
   for i in range(0,4):
      final_pred = np.zeros((8,3))
      final_pred = offset[i] + anchor_point[i, None]
      render_pcl(final_pred.T, name = str(i))
      render_pcl(gt_boxes[i].T, name = str(i))
   plt.show()

