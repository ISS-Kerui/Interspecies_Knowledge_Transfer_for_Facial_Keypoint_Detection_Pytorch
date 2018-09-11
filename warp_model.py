# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tps_grid_gen2 import TPSGridGen
import torchvision.models


class Alex(nn.Module):
    def __init__(self, num_output):
        super(Alex, self).__init__()
        self.features = torchvision.models.alexnet(pretrained=True).features
        self.add_cnn = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=1,groups=2),
            nn.BatchNorm2d(128)
        
        )
        
        self.fc1 = nn.Linear(128*6*6,128)
        self.fc2 =  nn.Linear(128, num_output)
            

    def forward(self, x):
        x = self.features(x)
        x = self.add_cnn(x)      
        x = x.view(x.size(0), 128*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = Alex(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = Alex(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class STNNet(nn.Module):

    def __init__(self, args):
        super(STNNet, self).__init__()
        self.args = args

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)
        target_control_points = torch.cat([X, Y], dim = 1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[args.model]
        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points)

        self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points)


    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.args.image_height, self.args.image_width, 2)
        
        
        return grid

def get_model(args):

  
    print('create model with STN')
    model = STNNet(args)
    return model
