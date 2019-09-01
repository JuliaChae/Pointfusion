import torch
import torch.nn as nn
import torch.nn.functional as F 
import pdb

from Pointnet import PointNetfeat, STN3d
import torchvision.models as models
from utils import ResNet50Bottom

class MLP_Global(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.pointNet = PointNetfeat(global_feat = True, feature_transform= False)
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv2 = ResNet50Bottom(self.res50_model)
        
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8*3) # 8*3 for 3D bounding box coordinates
        self.fc5 = nn.Linear(128, 11) # 10 classes 

    def forward(self, im, pts):
        batch_size = im.size()[0]
        npts= pts.size()[2]

        basefeat = self.res50_conv2(im).view(batch_size, 1, 2048)
        globalfeat, pointfeat, _ = self.pointNet(pts)

        basefeat = F.normalize(basefeat, p=2, dim=2)
        globalfeat = F.normalize(globalfeat, p=2, dim=2)
        
        fusionfeat = torch.FloatTensor(batch_size, 1, 3072).cuda()
        fusionfeat = torch.cat([globalfeat, basefeat], dim=2)
        
        x = fusionfeat
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        boxes = self.fc4(x)
        boxes = boxes.view(-1,8,3)
        classes = self.fc5(x)
        return boxes, classes

def weights_init(m):                                                                                                                                                                              
    classname = m.__class__.__name__                                                                      
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:                                                                                                                          
        try:                                                                                                
            nn.init.xavier_uniform_(m.weight.data)                                                                                                                                                      
            m.bias.data.fill_(0)                                                                                                                                                                                     
                                                                                                        
        except AttributeError:                                                                   
            print("Skipping initialization of ", classname) 

class MLP_Dense(nn.Module):
    def __init__(self):
        super(MLP_Dense, self).__init__()
        self.pointNet = PointNetfeat(global_feat = True, feature_transform= False)
        self.res50_model = models.resnet50(pretrained=True)
        self.res50_conv2 = ResNet50Bottom(self.res50_model)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8*3) # 8*3 for 3D bounding box coordinates
        self.fc5 = nn.Linear(128, 1) 

        self.apply(weights_init)

    def forward(self, im, pts):
        batch_size = im.size()[0]
        npts= pts.size()[2]

        basefeat = self.res50_conv2(im).view(batch_size, 1, 2048)
        globalfeat, pointfeat, _ = self.pointNet(pts)

        basefeat = F.normalize(basefeat, p=2, dim=2)
        globalfeat = F.normalize(globalfeat, p=2, dim=2)
        pointfeat = F.normalize(pointfeat, p=2, dim=2)

        basefeat = basefeat.repeat(1,npts,1)
        globalfeat = globalfeat.repeat(1,npts,1)
        
        # fusion
        fusionfeat = torch.cat((basefeat, globalfeat, pointfeat), 2)  # 180
        
        x = fusionfeat
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        offset = self.fc4(x)
        offset = offset.view(batch_size, npts,8,3)
        scores = self.fc5(x)
        scores = scores.view(-1, npts)
        
        # Shift scores so minimum is 0
        minimum = (scores.min(dim=1)[0]).view(batch_size,-1)
        scores = scores - minimum

        # Add eps to prevent returning 0 
        eps = 1e-4
        scores = scores + eps

        # Divide by range to normalize 
        s_range = scores.max(dim = 1)[0] - scores.min(dim=1)[0]
        s_range = s_range.view(batch_size,-1)
        scores = scores/s_range 
        
        return offset, scores
