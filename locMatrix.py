import os
import numpy as np
from tps_grid_gen import TPSGridGen
import torch
from grid_sample import grid_sample
import cv2
from torch.autograd import Variable
simi_dir = '../data/test_simi'
save_dir = '../data/test_matrix'
animal_npy = '../data/test_npy'
human_npy = '../data/npy2'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for txt_file in os.listdir(simi_dir):
	if txt_file.split('.')[-1] == 'txt':

		animal_name = txt_file.split('.')[0]
		txt_path = os.path.join(simi_dir,txt_file)
		f = open(txt_path,'r')
		line = f.read()
		human_names = line.split(' ')
		f.close()
		for i in range(len(human_names)):
			if human_names[i] != 'used':
			
				index = []
				source = np.expand_dims(np.load(animal_npy+'/'+animal_name+'.npy')[:,0:2],axis=0)
				target = np.load(human_npy+'/'+human_names[i]+'.npy')[:,0:2]
				for j in range(5):
					if source[0][j][0] <0 or target[j][0]<0:
						index.append(j)
						
				source = np.delete(source,index,axis=1)
				target = np.delete(target,index,axis=0)
				source = source/112.-1
				target = target/112.-1
				target_control_points = torch.Tensor(target)
				source_control_points = torch.Tensor(source)

				tps = TPSGridGen(224, 224, target_control_points)
				source_coordinate = tps(source_control_points)
				grid = source_coordinate.view(1, 224, 224, 2)
				np.save(save_dir+'/'+animal_name+'@'+human_names[i]+'.npy',grid)
				# try:
				# 	x = cv2.imread('../datahead/'+animal_name+'.jpg')
				# 	x = np.expand_dims(x,axis=0)
				# 	x = np.transpose(x,(0,3,1,2)) 
				# 	x = Variable(torch.Tensor(x))
				# except:
				# 	x = cv2.imread('../data/head/'+animal_name+'.JPG')
				# 	x = np.expand_dims(x,axis=0)
				# 	x = np.transpose(x,(0,3,1,2)) 
				# 	x = Variable(torch.Tensor(x))
				# transformed_x = grid_sample(x, grid)
				# transformed_x = np.transpose(transformed_x.data.numpy(),(0,2,3,1))
				# transformed_x = np.squeeze(transformed_x)
				# cv2.imwrite(save_dir+'/'+animal_name+'@'+human_names[i]+'.jpg',transformed_x)
			else:
				print "haha"