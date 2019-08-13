import torch
import torch.nn as nn
import torch.nn.functional as F 
import pdb


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        #self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8*3) # 8*3 for 3D bounding box coordinates
        self.fc5 = nn.Linear(128, 11) # 10 classes 

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        boxes = self.fc4(x)
        boxes = boxes.view(-1,8,3)
        classes = self.fc5(x)
        return boxes, classes

class MLP_Dense(nn.Module):
    def __init__(self):
        super(MLP_Dense, self).__init__()
        self.fc1 = nn.Linear(3136, 512)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        #self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8*3) # 8*3 for 3D bounding box coordinates
        self.fc5 = nn.Linear(128, 1) 

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        offset = self.fc4(x)
        offset = offset.view(-1,8,3)
        score = self.fc5(x)
        return offset, score
