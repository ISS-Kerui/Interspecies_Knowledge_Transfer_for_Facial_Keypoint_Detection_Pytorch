import argparse
import detectNet
import os
import numpy as np
import datagenerator as datagen
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

def eucldist_vectorized(coords1, coords2):
    """ Calculates the euclidean distance between 2 lists of coordinates. """
    return np.sqrt(np.sum((coords1 - coords2)**2))

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 40)
parser.add_argument('--test-batch-size', type = int, default = 300)
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--log-interval', type = int, default = 50)
parser.add_argument('--save-interval', type = int, default = 500)
parser.add_argument('--seed', type = int, default = 1)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = detectNet.Vanilla()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas=(0.9, 0.99))

data_dir = '/Users/zkr/Desktop/animal-human/warp-network/create-ground-truth/data/face'
label_dir ='/Users/zkr/Desktop/animal-human/warp-network/create-ground-truth/data/face_npy'
test_dir = '/Users/zkr/Desktop/animal-human/warp-network/create-ground-truth/data/face'
test_npy = '/Users/zkr/Desktop/animal-human/warp-network/create-ground-truth/data/face_npy'

def train(epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    gen = datagen.VanillaGenerator(data_dir=test_dir,npy_dir=test_npy,batch_size=args.batch_size,img_size=224)
    while gen.changed == False:
        
        data,label = gen.get()
        data = torch.Tensor(data)
        label = torch.Tensor(label)
        if args.cuda:
            data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.SmoothL1Loss()
        
        loss = criterion(output,label)
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

def eval(epoch):
	model.eval()
	model.load_state_dict(torch.load('../detect.pth'))
	gen = datagen.VanillaGenerator(data_dir=data_dir,npy_dir=label_dir,batch_size=args.batch_size,img_size=224)
	data,np_label = gen.get_test(test_dir,test_npy,args.test_batch_size)
	data = torch.Tensor(data)
	label = torch.Tensor(np_label)
	if args.cuda:
		data, label = data.cuda(), label.cuda()

	data, label = Variable(data), Variable(label)
	optimizer.zero_grad()
	output = model(data)
	criterion = nn.SmoothL1Loss()
	loss = criterion(output,label)
	if args.cuda:
		print str(epoch)+'   val_loss: '+str(loss.cpu().data.numpy()[0])
		np_output =output.cpu().data.numpy()
	else:
		print str(epoch)+'   val_loss: '+str(loss.data.numpy()[0])
		np_output =output.data.numpy()
	accuracy = [0,0,0,0,0]
	for i in range(args.batch_size):
			if eucldist_vectorized(np_output[i,0:2],np_label[i,0:2]) <1/10.:
				accuracy[0] += 1
			if eucldist_vectorized(np_output[i,2:4],np_label[i,2:4]) <1/10.:
				accuracy[1] += 1
			if eucldist_vectorized(np_output[i,4:6],np_label[i,4:6]) <1/10.:
				accuracy[2] += 1
			if eucldist_vectorized(np_output[i,6:8],np_label[i,6:8]) <1/10.:
				accuracy[3] += 1
			if eucldist_vectorized(np_output[i,8:],np_label[i,8:]) <1/10.:
				accuracy[4] += 1
	for i in range(5):
		accuracy[i] =accuracy[i] /(args.batch_size+0.0)
	print ('-----------------------------------------------')
	print ('the accuracy of left eye is '+str(accuracy[0]))
	print ('the accuracy of right eye is '+str(accuracy[1]))
	print ('the accuracy of nose is '+str(accuracy[2]))
	print ('the accuracy of left mouth is '+str(accuracy[3]))
	print ('the accuracy of right mouth is '+str(accuracy[4]))
	print ('-----------------------------------------------')




checkpoint_dir = 'output/checkpoint/model_weights/' 

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)



for epoch in range(1, args.epochs + 1):
	#train(epoch)
	eval(epoch)