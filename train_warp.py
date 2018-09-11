# encoding: utf-8

import os
import torch
import random
import argparse
import warp_model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import datagenerator as datagen
from grid_sample import grid_sample
import numpy as np
import cv2
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 30)
parser.add_argument('--test-batch-size', type = int, default = 1000)
parser.add_argument('--epochs', type = int, default = 40)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 30)
parser.add_argument('--save-interval', type = int, default = 100)
parser.add_argument('--model', required = True)
parser.add_argument('--angle', type = int, default=60)
parser.add_argument('--span_range', type = int, default = 0.9)
parser.add_argument('--grid_size', type = int, default = 5)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.span_range_height = args.span_range_width = args.span_range
args.grid_height = args.grid_width = args.grid_size
args.image_height = args.image_width = 224

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = warp_model.get_model(args)
if args.cuda:
    model.cuda()

pretrained_params = list(map(id, model.loc_net.cnn.features.parameters()))
other_params = filter(lambda p: id(p) not in pretrained_params, model.parameters())

optimizer = optim.Adam( [ {'params': other_params},
                          {'params': model.loc_net.cnn.features.parameters(), 'lr': args.lr/10.}]
            , lr = args.lr, betas=(0.9, 0.99))

data_dir = '/data/head'
loc_matrix_dir ='/data2/matrix'

def train(epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    gen = datagen.ImageDataGenerator(data_dir=data_dir,loc_matrix_dir=loc_matrix_dir,batch_size=args.batch_size,img_size=224)
    while gen.changed == False:
        
        data,target = gen.get()
        data = torch.Tensor(data)
        target = torch.Tensor(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.MSELoss()
        
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        if gen.step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, gen.step * len(data), gen.data_size,
                100. * gen.step * len(data)/ gen.data_size, loss.data[0]))
        if gen.step % args.save_interval == 0:
            checkpoint_path = checkpoint_dir + 'epoch%03d_iter%03d.pth' % (epoch, gen.step)
            torch.save(model.cpu().state_dict(), checkpoint_path)
            if args.cuda:
                model.cuda()

def eval(path,save_dir):
    model.load_state_dict(torch.load(path))
    model.eval()
    gen = datagen.ImageDataGenerator(data_dir=data_dir,loc_matrix_dir=loc_matrix_dir,batch_size=args.batch_size,img_size=224)
    while gen.changed == False:
        
        data,target = gen.get()
        data = torch.Tensor(data)
        target = torch.Tensor(target)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        transformed_x = grid_sample(data*255., output)
        out_pics = np.transpose(transformed_x.data.numpy(),(0,2,3,1))
        for pic in out_pics:
            cv2.imwrite(save_dir+'/'+str(gen.cursor)+'.jpg',pic)



checkpoint_dir = '/output/checkpoint/%s_angle%d_grid%d/' % (
    args.model, args.angle, args.grid_size,
)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)



for epoch in range(1, args.epochs + 1):
    train(epoch)

# eval('../epoch040_iter100.pth','img')