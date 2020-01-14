import torch
import torch.nn as nn
import torch.nn.functional as F

def SoftClamp(x, n):
    slow = 10
    return (torch.min(F.leaky_relu(x, 2/(n/slow+1)), torch.ones_like(x)) - torch.min(F.leaky_relu(1-x, 2/(n/slow+1)), torch.ones_like(x))+1)/2.0

def SoftLeaky(x, n):
	slow = 10
	return F.leaky_relu(x, 1/(n/slow+1))



class Model(nn.Module):

	def __init__(self, width, features = 9):
		super().__init__()

		self.width = width
		self.features = features
		self.DO = 0.5
		self.iter = 0

		self.trunk = nn.Sequential(
			nn.Linear(self.features, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO),
			nn.Linear(self.width, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO),
			nn.Linear(self.width, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO),
			nn.Linear(self.width, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO))

		self.width //= 2

		self.pred = nn.Sequential(
			nn.Linear(2*self.width, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO),
			nn.Linear(self.width, self.width//2), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width//2), nn.Dropout(self.DO),
			nn.Linear(self.width//2, self.width//4), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width//4),
			nn.Linear(self.width//4, 1))

		self.prob = nn.Sequential(
			nn.Linear(2*self.width, self.width), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width), nn.Dropout(self.DO),
			nn.Linear(self.width, self.width//2), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width//2), nn.Dropout(self.DO),
			nn.Linear(self.width//2, self.width//4), nn.ReLU(inplace=True), nn.BatchNorm1d(self.width//4),
			nn.Linear(self.width//4, 1))

		self.width *= 2

	def forward(self, x):

		x = self.trunk(x)

		#prob = SoftClamp(self.prob(x), self.iter)
		prob = torch.sigmoid(self.prob(x))[:,0]
		#prob = self.prob(x)[:,0].clamp(0,1)
		pred = SoftLeaky(self.pred(x), self.iter)[:,0]

		self.iter += 1

		return prob, pred


	def L1_Loss(self):
		L1_loss = 0
		for param in self.parameters():
			L1_loss += torch.sum(torch.abs(param))
		return L1_loss
