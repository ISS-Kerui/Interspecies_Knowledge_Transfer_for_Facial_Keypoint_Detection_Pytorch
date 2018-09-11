import math
import numpy as np
import os
import copy
import tensorflow as tf
def _distance(points):
	dis1 = np.square(points[0][0]-points[2][0])+np.square(points[0][1]-points[2][1])
	dis2 = np.square(points[1][0]-points[2][0])+np.square(points[1][1]-points[2][1])
	trend = 'null'
	if dis2-dis1 >10:
		trend = 'left'
	elif dis1-dis2 > 10:
		trend = 'right'
	else:
		trend = 'none'
	return trend



def _angle1(points):
	if points[1][0] > points[0][0]:
		midPoint_ay = (points[1][0]-points[0][0])/2.+points[0][0]
	else:
		midPoint_ay = (points[0][0]-points[1][0])/2.+points[1][0]
	if points[1][1] > points[0][1]:
		midPoint_ax = (points[1][1]-points[0][1])/2.+points[0][1]
	else:
		midPoint_ax = (points[0][1]-points[1][1])/2.+points[1][1]
	
	k1 = (midPoint_ay-points[2][0])/((midPoint_ax-points[2][1])+0.1)
	
	k2 = -1/((points[1][0]-points[0][0])/(points[1][1]-points[0][1]+0.1))
	
	if k1 >0:
		trend = 'right'
	else:
		trend = 'left'

	Cobb =float(math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5)
	
	return Cobb,trend
def _angle2(points):
	k1 = (points[0][0]-points[2][0])/(points[0][1]-points[2][1])
	k2 = (points[3][0]-points[2][0])/(points[3][1]-points[2][1])
	Cobb =float(math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5)
	
	
	return Cobb

def _angle3(points):
	k1 = (points[1][0]-points[2][0])/(points[1][1]-points[2][1])
	k2 = (points[4][0]-points[2][0])/(points[4][1]-points[2][1])
	Cobb =float(math.fabs(np.arctan((k1-k2)/(float(1 + k1*k2)))*180/np.pi)+0.5)
	
	return Cobb

animal_points = []
animal_names = []
human_points = []
human_names = []
save_dir = 'simi'
K = 5
for npy_file in os.listdir('../data/test_npy'):
	path = os.path.join('../data/test_npy',npy_file)
	animal_point = np.load(path)
	animal_names.append(npy_file.split('.')[0])
	animal_points.append(animal_point)
for npy_file in os.listdir('../data/npy2'):
	path = os.path.join('../data/npy2',npy_file)
	human_point = np.load(path)
	human_names.append(npy_file.split('.')[0])
	human_points.append(human_point)
l = 0
for al in animal_points:
	copy_hnames = copy.deepcopy(human_names)
	if al[0][2] == 1 and al[1][2] ==1 and al[2][2]==1:
		aangle,atrend= _angle1(al)
		selected_names = []
		for iter_num in range(0,K):
			min_ds = 360
			index = 0
			for i in range(len(human_points)):

				if human_points[i][0][2] == 1 and human_points[i][1][2] ==1 and human_points[i][2][2]==1:
					if copy_hnames[i] != 'used':
						hangle,htrend= _angle1(human_points[i])
						if abs(aangle-hangle)<min_ds and (atrend == htrend):
							min_ds = abs(aangle-hangle)
							index = i
			selected_names.append(copy_hnames[index])
			copy_hnames[index] ='used'
		row = selected_names[0]+' '+ selected_names[1]+' '+selected_names[2]+' '+selected_names[3]+' '+selected_names[4]
		txtName = animal_names[l]+'.txt'
		save_path = os.path.join(save_dir,txtName)
		f=file(save_path, "w+")
		f.write(row)
		f.close()




	
	elif al[0][2] == 1  and al[2][2] ==1 and al[3][2]==1:
		
		aangle = _angle2(al)
		selected_names = []
		for iter_num in range(0,K):
			min_ds = 360
			for i in range(len(human_points)):
				if human_points[i][0][2] == 1 and human_points[i][2][2] ==1 and human_points[i][3][2]==1:
					if human_points[i][1][2] == 1:
						trend = _distance(human_points[i])
						if copy_hnames[i] != 'used':
							hangle= _angle2(human_points[i])
							if abs(aangle-hangle)<min_ds and trend == 'right':
								min_ds = abs(aangle-hangle)
								index = i
					else:
						if copy_hnames[i] != 'used':
							hangle= _angle2(human_points[i])
							if abs(aangle-hangle)<min_ds:
								min_ds = abs(aangle-hangle)
								index = i
			selected_names.append(copy_hnames[index])
			copy_hnames[index] ='used'
		row = selected_names[0]+' '+ selected_names[1]+' '+selected_names[2]+' '+selected_names[3]+' '+selected_names[4]
		txtName = animal_names[l]+'.txt'
		save_path = os.path.join(save_dir,txtName)
		f=file(save_path, "w+")
		f.write(row)
		f.close()

	elif al[1][2] == 1  and al[2][2] ==1  and al[4][2]==1:
	
		aangle= _angle3(al)
		selected_names = []
		for iter_num in range(0,K):
			min_ds = 360
			for i in range(len(human_points)):
				if human_points[i][1][2] == 1 and human_points[i][2][2] ==1 and human_points[i][4][2]==1:
					if human_points[i][0][2] == 1:
						trend = _distance(human_points[i])
						if copy_hnames[i] != 'used':
							hangle= _angle3(human_points[i])
							if abs(aangle-hangle)<min_ds and trend =='left':
								min_ds = abs(aangle-hangle)
								index = i
					else:
						if copy_hnames[i] != 'used':
							hangle= _angle3(human_points[i])
							if abs(aangle-hangle)<min_ds:
								min_ds = abs(aangle-hangle)
								index = i

			selected_names.append(copy_hnames[index])
			copy_hnames[index] ='used'
		row = selected_names[0]+' '+ selected_names[1]+' '+selected_names[2]+' '+selected_names[3]+' '+selected_names[4]
		txtName = animal_names[l]+'.txt'
		save_path = os.path.join(save_dir,txtName)
		f=file(save_path, "w+")
		f.write(row)
		f.close()
	l = l +1