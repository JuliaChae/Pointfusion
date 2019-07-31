import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.multiprocessing as mp 
import torchvision.models as models
import argparse 

from MLP import MLP 
from utils import ResNet50Bottom
from Pointnet import PointNetfeat, STN3d
import pdb
from train import train

# Training settings
parser = argparse.ArgumentParser(description='Pointfusion Multithreading')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')

if __name__ == '__main__':
   args = parser.parse_args()
   torch.manual_seed(1)
   np.set_start_method('spawn')
   model = MLP().cuda()
   model.share_memory()

   res50_model = models.resnet50(pretrained=True).cuda()
   res50_model.share_memory()
   res50_model.eval()
   res50_conv2 = ResNet50Bottom(res50_model)

   pointfeat = PointNetfeat(global_feat=True).cuda()
   pointfeat.share_memory()
   pointfeat.eval()

   trans = STN3d().cuda()
   trans.share_memory()
   trans.eval()

   processes = []
   for rank in range(args.num_processes):
      p = mp.Process(target=train, args=(rank, model, res50_model, pointfeat, trans, args.epochs))
      p.start()
      processes.append(p)
      print("Rank: " + str(rank))
   for p in processes:
      p.join()