

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import os
import cv2
import random
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

np.set_printoptions(threshold=np.inf)  
class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, data_dir,loc_matrix_dir,batch_size,img_size,shuffle=True):
        """Create a new ImageDataGenerator.
        """
        self.data_dir = data_dir
        self.loc_matrix_dir = loc_matrix_dir
        self.batch_size = batch_size
        self.image_size = img_size
        self.data_size = 0
        # retrieve the data from the text file
        self.step = 0
        self.cursor = 0
        # number of samples in the dataset
        self.shuffle =shuffle
        self._get_train_img_path()
        self.changed = False
        self._shuffle_lists()

    def _get_train_img_path(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        
        for img_name in os.listdir(self.loc_matrix_dir):
            if img_name.split('.')[-1] == 'npy':
                name = img_name.split('.')[0]
                self.img_paths.append(name)
        self.data_size = len(self.img_paths)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        for i in permutation:
            self.img_paths.append(path[i])
            

    def get(self):
        """create data for Alexnet   """
        self.step = self.step + 1
        images = np.zeros(
            (self.batch_size, 3,self.image_size, self.image_size))
        label_matrix = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 2))
        count = 0
        while count < self.batch_size:

            pair = self.img_paths[self.cursor]
            imname = self.data_dir+'/'+pair.split('@')[0]+'.jpg'
            images[count, :, :, :] = self._image_read(imname)
            matrix = np.load(self.loc_matrix_dir+'/'+ self.img_paths[self.cursor]+'.npy')
            label_matrix[count,:,:,:] = matrix


            count += 1
            self.cursor += 1
            if self.cursor >= self.data_size:
                np.random.shuffle(self.img_paths)
                self.cursor = 0
                self.changed = True

            #     self.epoch += 1
    
        return images,label_matrix

    def get_all(self,animal_npy):

        """create data for Alexnet   """
        self.step = self.step + 1
        images = np.zeros(
            (self.batch_size, 3,self.image_size, self.image_size))
        label_matrix = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 2))
        labels = np.zeros(
            (self.batch_size,5,2))
        count = 0
        while count < self.batch_size:

            pair = self.img_paths[self.cursor]
            self.imname = self.data_dir+'/'+pair.split('@')[0]+'.jpg'
            npyname = animal_npy+'/'+ pair.split('@')[0]+'.npy'
            images[count, :, :, :] = self._image_read(self.imname)
            labels[count,:,:] =  self._npy_read(npyname)
            matrix = np.load(self.loc_matrix_dir+'/'+ self.img_paths[self.cursor]+'.npy')
            label_matrix[count,:,:,:] = matrix


            count += 1
            self.cursor += 1
            if self.cursor >= self.data_size:
                np.random.shuffle(self.img_paths)
                self.cursor = 0
                self.changed = True

            #     self.epoch += 1
    
        return images,label_matrix,labels


    def get_test_all(self,test_dir,test_npy):
        
        pic_list = []
        labels_list = []
        pics = []
        labels = []
        for pic_name in os.listdir(test_dir):
            if pic_name.split('.')[-1] == 'jpg' or pic_name.split('.')[-1] == 'JPG':
                pic_list.append(test_dir+'/'+pic_name)
                labels_list.append(test_npy+'/'+pic_name.split('.')[0]+'.npy')
        for pic in pic_list:
            pics.append(self._image_read(pic))
        for label in labels_list:
            labels.append(self._npy_read2(label))
        return np.array(pics),np.array(labels).reshape(-1,10)

    def _image_read(self, imname):
        try:
            image = cv2.imread(imname)
            #image = cv2.resize(image, (self.image_size, self.image_size))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image / 255.0)
        except:
            image = cv2.imread(imname.split('.')[0]+'.JPG')
            #image = cv2.resize(image, (self.image_size, self.image_size))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image = (image / 255.0)
        image = np.transpose(image,(2,0,1)) 
        return image

    def _npy_read(self,npyname):
        npy = np.load(npyname)
        npy = npy[:,0:2]/112.-1
        
        return npy

    def _npy_read2(self,npyname):
        npy = np.load(npyname)
        npy = npy[:,0:2]/224.
        
        return npy
class VanillaGenerator(object):
    """Vanilla class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, data_dir,npy_dir,batch_size,img_size,shuffle=True):
        """Create a new ImageDataGenerator.
        """
        self.data_dir = data_dir
        self.npy_dir = npy_dir
        self.batch_size = batch_size
        self.image_size = img_size
        self.data_size = 0
        # retrieve the data from the text file
        self.step = 0
        self.cursor = 0
        # number of samples in the dataset
        self.shuffle =shuffle
        self._get_train_img_path()
        self.changed = False
        self._shuffle_lists()

    def _get_train_img_path(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.npy_paths = []
        for img_name in os.listdir(self.data_dir):
            if img_name.split('.')[-1] == 'jpg':
                img_path = os.path.join(self.data_dir,img_name)
                self.img_paths.append(img_path)
                npy_path = os.path.join(self.npy_dir,img_name.split('.')[0]+'.npy')
                self.npy_paths.append(npy_path)
        self.data_size = len(self.img_paths)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        paths = self.img_paths
        new_npy_paths = self.npy_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.npy_paths = []
        for i in permutation:
            self.img_paths.append(paths[i])
            self.npy_paths.append(new_npy_paths[i])
            

    def get(self):
        """create data for Alexnet   """
        self.step = self.step + 1
        images = np.zeros(
            (self.batch_size, 3,self.image_size, self.image_size))
        labels = np.zeros(
            (self.batch_size,10))
        count = 0
        while count < self.batch_size:

            images[count, :, :, :] = self._image_read(self.img_paths[self.cursor])
            labels[count,:] = self._npy_read(self.npy_paths[self.cursor])

            count += 1
            self.cursor += 1
            if self.cursor >= self.data_size:
                np.random.shuffle(self.img_paths)
                self.cursor = 0
                self.changed = True

            #     self.epoch += 1
    
        return images,labels


    def get_test(self,test_dir,test_npy,test_batch_size):
        pic_list = []
        npy_list = []
        pics = []
        npys = []
        for pic_name in os.listdir(test_dir):
            pic_path = os.path.join(test_dir,pic_name)
            pic_list.append(pic_path)
            # npy_path = os.path.join(test_npy,pic_name.split('.')[0]+'.npy')
            # npy_list.append(npy_path)
        numbers = random_int_list(0,len(pic_list),test_batch_size)
        random_pic_list = [pic_list[number] for number in numbers]
        
        for pic_path in random_pic_list:
    
            npy_path = os.path.join(test_npy, pic_path.split('/')[-1].split('.')[0]+'.npy')
            npy_list.append(npy_path)
        for pic in random_pic_list:
            pics.append(self._image_read(pic))
        for n in npy_list:
            npys.append(self._npy_read(n))
        return np.array(pics),np.squeeze(np.array(npys))

    def _image_read(self, imname):
        try:
            image = cv2.imread(imname)
            image = (image / 255.0)
        except:
            image = cv2.imread(imname.split('.')[0]+'.JPG')
            image = (image / 255.0)
        image = np.transpose(image,(2,0,1)) 
        return image

    def _npy_read(self,npyname):
        npy = np.load(npyname)
        npy = npy[:,0:2]
        npy = np.reshape(npy,[-1,10])/(self.image_size+0.0)
        return npy

