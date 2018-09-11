import os
import torch
import random
import argparse
import warp_model
import detectNet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import datagenerator as datagen
from grid_sample import grid_sample
import numpy as np
import cv2
import copy
from tensorboardX import SummaryWriter

#warp_model_weights = '../warp.pth'
#detect_model_weights = '../detect.pth'
weights = '../epoch38.pth'
def eucldist_vectorized(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((coords1 - coords2)**2))

class AnimalNet(nn.Module):

	def __init__(self, args):
		super(AnimalNet, self).__init__()
		self.args = args
		self.warp_model = warp_model.get_model(args)
		#self.warp_model.load_state_dict(torch.load(warp_model_weights))
		self.detect_model = detectNet.Vanilla()
		#self.detect_model.load_state_dict(torch.load(detect_model_weights))
	def forward(self, x):
		#batch_size = x.size(0)
		grid = self.warp_model(x)
		x = grid_sample(x*255., grid)
		x = self.detect_model(x/255.)
		return x

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 30)
parser.add_argument('--test-batch-size', type = int, default = 60)
parser.add_argument('--epochs', type = int, default = 25)
parser.add_argument('--lr', type = float, default = 0.000002)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 30)
parser.add_argument('--save-interval', type = int, default = 200)
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


model = AnimalNet(args)
model.load_state_dict(torch.load(weights))
if args.cuda:
	print "cuda success"
	model.cuda()

pretrained_params = list(map(id, model.warp_model.loc_net.cnn.features.parameters()))
detect_params = list(map(id, model.detect_model.parameters()))
other_params = filter(lambda p: id(p) not in (pretrained_params+detect_params), model.parameters())

optimizer = optim.Adam( [ {'params': other_params},
                          {'params': model.warp_model.loc_net.cnn.features.parameters(), 'lr': args.lr/10.},
                          {'params': model.detect_model.parameters(), 'lr': args.lr*10.}]
            , lr = args.lr, betas=(0.9, 0.99))

data_dir = '../data/head'
loc_matrix_dir ='../data/matrix'
animal_label_dir ='../data/npy'
test_dir = '../data/test'
# test_matrix_dir = '../data/test_matrix'
test_label_dir = '../data/test_npy'

writer = SummaryWriter()
def train(epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    gen = datagen.ImageDataGenerator(data_dir=data_dir,loc_matrix_dir=loc_matrix_dir,batch_size=args.batch_size,img_size=224)
    sum_loss1 = []
    sum_loss2 = []
    while gen.changed == False:
        
        data,target,label= gen.get_all(animal_label_dir)
        data = torch.Tensor(data)
        target = torch.Tensor(target)
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target= Variable(data), Variable(target)
        optimizer.zero_grad()
        matrix = model.warp_model(data)
        if args.cuda:
            t_matrix = matrix.cpu().data.numpy().reshape(args.batch_size,50176,2)
        else:
            t_matrix = matrix.data.numpy().reshape(args.batch_size,50176,2)
        

        for i in range(args.batch_size):
            #n_matrix = copy.deepcopy(t_matrix)
            # matrix_one = copy.deepcopy(t_matrix[i])
            for j in range(5):
                matrix_one = copy.deepcopy(t_matrix[i])
           
                if (label[i,j,0]+1)>0:
         
                   
                    matrix_one[:,0] = matrix_one[:,0]-label[i,j,0]
                    matrix_one[:,1] = matrix_one[:,1]-label[i,j,1]
                    matrix_one = np.abs(matrix_one)
                    matrix_one = np.sum(matrix_one,axis=-1)
                    index = np.argmin(matrix_one)
                    
                    label[i,j,:] = np.array([(index-index/224*224)/224.,(index/224)/224.])
                else:
                    label[i,j,:] = ((label[i,j,:]+1)*112)/224.
        
        
        label = label.reshape(args.batch_size,10)

        #----------print pics test -----------

     #    data2 = np.transpose(data,[0,2,3,1])*255.
     #    data2 = np.transpose(data2,[0,3,1,2])
     #    data2 = torch.Tensor(data2)
     #    if args.cuda:
     #        data = data.cuda()
     #        data2 = data.cuda()

     #    data= Variable(data)
     #    data2=Variable(data2)
     #    grid = model.warp_model(data)
     #    pics = grid_sample(data2, grid)
     #    output = model(data)

     #    if args.cuda:
     #        pics = np.transpose(pics.cpu().data.numpy(),(0,2,3,1))
		    
		
     #    else:
       
	        
     #        pics = np.transpose(pics.data.numpy(),(0,2,3,1))
	
	    # #print pics.shape
     #    points = label.reshape([-1,5,2])
     #    points = points.astype(np.int32)

     #    for i in range(args.batch_size):
     #        image = pics[i]*255.
     #        cv2.imwrite('result.jpg',image)
     #        image = cv2.imread('result.jpg')/255.
		
     #        for j in range(5):
			
     #            cv2.circle(image, center = tuple(points[i][j]), radius= 4, color= (255,255,0), thickness= -1)
     #        cv2.imshow('result.jpg',image)
     #        cv2.waitKey(0)






         #---------------------------------------
        
        label = torch.Tensor(label)
        if args.cuda:
            label = label.cuda()
        label = Variable(label)
        output = model(data)
        criterion1 = nn.MSELoss()
        criterion2 = nn.SmoothL1Loss()
        loss1 = criterion1(matrix,target)
        sum_loss1.append(loss1)
        loss1.backward()
        loss2 = criterion2(output,label)
        sum_loss2.append(loss2)
        loss2.backward()
        optimizer.step()
        
        if gen.step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, gen.step * len(data), gen.data_size,
                100. * gen.step * len(data)/ gen.data_size, loss2.data[0]))
            if args.cuda:
          
                np_output =output.cpu().data.numpy()
                np_label = label.cpu().data.numpy()
            else:
       
                np_output =output.data.numpy()
                np_label = label.data.numpy()
            accuracy = [0,0,0,0,0]
            for i in range(args.batch_size):
                if eucldist_vectorized(np_output[i,0:2],np_label[i,0:2]) <1/20.:
                    accuracy[0] += 1
                if eucldist_vectorized(np_output[i,2:4],np_label[i,2:4]) <1/20.:
                    accuracy[1] += 1
                if eucldist_vectorized(np_output[i,4:6],np_label[i,4:6]) <1/20.:
                    accuracy[2] += 1
                if eucldist_vectorized(np_output[i,6:8],np_label[i,6:8]) <1/20.:
                    accuracy[3] += 1
                if eucldist_vectorized(np_output[i,8:],np_label[i,8:]) <1/20.:
                    accuracy[4] += 1
            for i in range(5):
                accuracy[i] =accuracy[i] /(args.batch_size+0.0)
            writer.add_text('train_accuracy','the accuracy of left eye is '+str(accuracy[0])+'.  '+'the accuracy of right eye is '+str(accuracy[1]) \
            +'.  '+'the accuracy of nose is '+str(accuracy[2])+'.  '+ 'the accuracy of left mouth is '+str(accuracy[3])+'.  '+'the accuracy of right mouth is '+str(accuracy[4]), gen.step)
            print ('-----------------------------------------------')
            print ('the accuracy of left eye is '+str(accuracy[0]))
            print ('the accuracy of right eye is '+str(accuracy[1]))
            print ('the accuracy of nose is '+str(accuracy[2]))
            print ('the accuracy of left mouth is '+str(accuracy[3]))
            print ('the accuracy of right mouth is '+str(accuracy[4]))
            print ('-----------------------------------------------')
        if gen.step % args.save_interval == 0:
            checkpoint_path = checkpoint_dir + 'epoch%03d_iter%03d.pth' % (epoch, gen.step)
            torch.save(model.cpu().state_dict(), checkpoint_path)
            if args.cuda:
                model.cuda()
    return sum(sum_loss1) / len(sum_loss1),sum(sum_loss2) / len(sum_loss2)


def test(save_dir):
  model.eval()
  gen = datagen.ImageDataGenerator(data_dir=data_dir,loc_matrix_dir=loc_matrix_dir,batch_size=args.test_batch_size,img_size=224)
  data,labels = gen.get_test_all(test_dir,test_label_dir)
  data2 = np.transpose(data,[0,2,3,1])
  pics = data2*255

  data = torch.Tensor(data)
  if args.cuda:
    data = data.cuda()
    # data2 = data.cuda()

  data= Variable(data)
  matrix = model.warp_model(data)
	# pics = grid_sample(data2, matrix)
  output = model(data)

  if args.cuda:
		# pics = np.transpose(pics.cpu().data.numpy(),(0,2,3,1))
    np_output =output.cpu().data.numpy()
    t_matrix = matrix.cpu().data.numpy()
  else:
       
    np_output =output.data.numpy()
		# pics = np.transpose(pics.data.numpy(),(0,2,3,1))
    t_matrix = matrix.data.numpy()


	
  points = np_output.reshape([-1,5,2])*224.

  transform_points = copy.deepcopy(points)
  for i in range(len(points)):
		  for j in range(5):
			    if points[i][j][0]>0:
				      transform_points[i][j] = t_matrix[i][int(round(points[i,j,1]))][int(round(points[i,j,0]))]
				      transform_points[i][j] = (transform_points[i][j]+1)*112
			    else:
				      transform_points[i][j] = points[i][j]
  predict_labels = transform_points.reshape(len(points),10)

  accuracy = [0,0,0,0,0]
  for i in range(len(points)):

    if eucldist_vectorized(predict_labels[i,0:2]/224.,labels[i,0:2]) <1/20.:
        accuracy[0] += 1
    if eucldist_vectorized(predict_labels[i,2:4]/224.,labels[i,2:4]) <1/20.:
        accuracy[1] += 1
    if eucldist_vectorized(predict_labels[i,4:6]/224.,labels[i,4:6]) <1/20.:
        accuracy[2] += 1
    if eucldist_vectorized(predict_labels[i,6:8]/224.,labels[i,6:8]) <1/20.:
        accuracy[3] += 1
    if eucldist_vectorized(predict_labels[i,8:]/224.,labels[i,8:]) <1/20.:
        accuracy[4] += 1
    if (predict_labels[i,0:2]/224.).any()<0 and labels[i,0:2].any()<0:
        accuracy[0] += 1
    if (predict_labels[i,2:4]/224.).any()<0 and labels[i,2:4].any()<0:
        accuracy[1] += 1
    if (predict_labels[i,4:6]/224.).any()<0 and labels[i,4:6].any()<0:
        accuracy[2] += 1
    if (predict_labels[i,6:8]/224.).any()<0 and labels[i,6:8].any()<0:
        accuracy[3] += 1
    if (predict_labels[i,8:]/224.).any()<0 and labels[i,8:].any()<0:
        accuracy[4] += 1


  for i in range(5):
      accuracy[i] =accuracy[i] /(len(points)+0.0)
  print ('-----------------------------------------------')
  print ('the accuracy of left eye is '+str(accuracy[0]))
  print ('the accuracy of right eye is '+str(accuracy[1]))
  print ('the accuracy of nose is '+str(accuracy[2]))
  print ('the accuracy of left mouth is '+str(accuracy[3]))
  print ('the accuracy of right mouth is '+str(accuracy[4]))
  print ('-----------------------------------------------')

  for i in range(len(pics)):
		  cv2.imwrite('result.jpg',pics[i])
		  pic = cv2.imread('result.jpg')
		  for j in range(5):
			  if transform_points[i][j][0]>0:
				  cv2.circle(pic, center = tuple(transform_points[i][j]), radius= 4, color= (255,255,0), thickness= -1)
		  cv2.imwrite(save_dir+'/'+'predict_'+str(i)+'.jpg',pic)
		


	
 #----------------no return to the original image ---------------
	# data2 = np.transpose(data,[0,2,3,1])*255.
	# data2 = np.transpose(data2,[0,3,1,2])
	# data2 = torch.Tensor(data2)
	# if args.cuda:
	# 	data = data.cuda()
	# 	data2 = data.cuda()

	# data= Variable(data)
	# data2=Variable(data2)
	# grid = model.warp_model(data)
	# pics = grid_sample(data2, grid)
	# output = model(data)

	# if args.cuda:
	# 	pics = np.transpose(pics.cpu().data.numpy(),(0,2,3,1))
	# 	np_output =output.cpu().data.numpy()
		
	# else:
       
	# 	np_output =output.data.numpy()
	# 	pics = np.transpose(pics.data.numpy(),(0,2,3,1))
	
	# #print pics.shape
	# print np,output
	# points = (np_output*224).astype(np.int32)
	# points = points.reshape(args.test_batch_size,5,2)

	# for i in range(args.test_batch_size):
	# 	image = pics[i]*255.
	# 	cv2.imwrite('result.jpg',image)
	# 	image = cv2.imread('result.jpg')/255.
	# 	print points
	# 	for j in range(5):
			
	# 		cv2.circle(image, center = tuple(points[i][j]), radius= 4, color= (255,255,0), thickness= -1)
	# 	cv2.imshow('result.jpg',image)
	# 	cv2.waitKey(0)

checkpoint_dir = 'output/checkpoint/animal_model_weights/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)



for epoch in range(1, args.epochs + 1):
    loss1,loss2 = train(epoch)
    writer.add_scalar('Train_warpNet_loss', loss1, epoch)
    writer.add_scalar('Train_detectNet_loss', loss2, epoch)
writer.close()

#test('../data/test_pic')