import torch

import torchvision
import torchvision.transforms as transforms
from networkFolder.ClassificationNetworkOnly import Net as classificationNetwork
from networkFolder.network import Net as reconstructionNetwork
import numpy as np
import random

num_classes = 10
batch_size = 1000

IMG_PATH = 'dataLinearMasks/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'train.csv'

class Map():
    def __init__(self, k):
        transform1=transforms.Compose([transforms.ToTensor()])
        self.test_dataset = torchvision.datasets.MNIST(root='MnistData/',
                                                        train = False,
                                                        transform=transform1,
                                                        download=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False)

        self.imageNumber = 0
        self.mnist = next(iter(self.test_loader))
        self.map = np.reshape(self.mnist[0][self.imageNumber].numpy(),(28,28))*255
        self.number = self.mnist[1][self.imageNumber].numpy()

        self.getNewMap(k)



    def getNewMap(self, k):
        self.imageNumber = self.imageNumber + k
        self.map = np.reshape(self.mnist[0][self.imageNumber].numpy(),(28,28))*255
        self.number = self.mnist[1][self.imageNumber].numpy()


class WorldEstimatingNetwork():
    def __init__(self):
        self.network  = reconstructionNetwork()
        self.network.load_state_dict(torch.load('networkFolder/class28x28reconstructionWeights', map_location={'cuda:0': 'cpu'}))

    def runNetwork(self,map,exploredArea):
        map = map * exploredArea
        map = torch.from_numpy(np.reshape(map/255.0,(1,1,28,28))).type(torch.FloatTensor)
        mask = torch.from_numpy(np.reshape(exploredArea,(1,1,28,28))).type(torch.FloatTensor)
        tensorReturn = self.network(map,mask)
        return np.reshape(tensorReturn.detach().numpy(),(28,28))*255

class DigitClassifcationNetwork():
    def __init__(self):
        self.network = classificationNetwork()
        self.network.load_state_dict(torch.load('networkFolder/class28x28classificationWeights', map_location={'cuda:0': 'cpu'}))

    def runNetwork(self,map):
        map = torch.from_numpy(np.reshape(map/255.0,(1,1,28,28)))
        tensorReturn = self.network(map)
        return tensorReturn.detach().numpy()


