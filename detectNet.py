import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Vanilla(nn.Module):
	def __init__(self):
		super(Vanilla, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3,16,kernel_size=5,stride=1),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.BatchNorm2d(16),

			nn.Conv2d(16,48,kernel_size=3,stride=1),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.BatchNorm2d(48),

			nn.Conv2d(48,64,kernel_size=3,stride=1),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.BatchNorm2d(64),

			nn.Conv2d(64,64,kernel_size=3,stride=1),
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.BatchNorm2d(64),

			nn.Conv2d(64,16,kernel_size=2,stride=1),
			nn.MaxPool2d(kernel_size=3,stride=2),
			nn.BatchNorm2d(16)

			)
		
		self.fc1 = nn.Linear(256,100)
		self.fc2 = nn.Linear(100, 10)

			
	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 256)
		x = F.tanh(self.fc1(x))

		x = F.tanh(self.fc2(x))
		return x