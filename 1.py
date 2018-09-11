import itertools
import numpy as np
import torch
from grid_sample import grid_sample
from tps_grid_gen import TPSGridGen
import cv2
from torch.autograd import Variable
import copy
source = np.expand_dims(np.load('IMG_3922.npy')[:,0:2]/112.-1,axis=0)
target = np.load('_-20_11_0.npy')[:,0:2]/112.-1

#print source
print (target+1)*112/224.

target_control_points = torch.Tensor(target)
source_control_points = torch.Tensor(source)


# Y, X = target_control_points.split(1, dim = 1)
# target_control_points = torch.cat([X, Y], dim = 1)
tps = TPSGridGen(224, 224, target_control_points)
source_coordinate = tps(source_control_points)
grid = source_coordinate.view(1, 224, 224, 2)
arr = grid.numpy()
arr1 = arr[0].reshape(-1,2)
for j in range(5):
	matrix_one = copy.deepcopy(arr1)
	
	if (source[0,j,0]+1)>0:

       
		matrix_one[:,0] = matrix_one[:,0]-source[0,j,0]
		matrix_one[:,1] = matrix_one[:,1]-source[0,j,1]
		matrix_one = np.abs(matrix_one)
		matrix_one = np.sum(matrix_one,axis=-1)
		index = np.argmin(matrix_one)

		source[0,j,:] = np.array([(index-index//224*224)/224.,index//224/224.])
	else:
		source[0,j,:] = (source[0,j,:]+1)*112./224.
print source

# ---transform points------------
# min_value = 4
# index1 = 0
# index2 = 0
# for x in range(224):
# 	for y in range(224):
		
# 		if (np.abs(arr[0,x,y,0]-(0.73214286))+np.abs(arr[0,x,y,1]-(-0.22321429)))<min_value:
# 			min_value = (np.abs(arr[0,x,y,0]-(0.73214286))+np.abs(arr[0,x,y,1]-(-0.22321429)))
# 			index1 = x
# 			index2 = y
# # print index1
# # print index2

# print min_value

# ----------------------




# x = cv2.imread('IMG_3922.JPG')
# x = np.expand_dims(x,axis=0)
# x = np.transpose(x,(0,3,1,2)) 
# x = Variable(torch.Tensor(x))
# transformed_x = grid_sample(x, grid)
# transformed_x = np.transpose(transformed_x.data.numpy(),(0,2,3,1))
# transformed_x = np.squeeze(transformed_x)
# cv2.imwrite('haha.jpg',transformed_x)