import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1, 0)
        self.conv2 = nn.Conv2d(20, 50, 5, 1, 0)

        self.Relu = nn.ReLU()
        self.linear1 = nn.Linear(5000,500)
        self.linear2 = nn.Linear(500,10)

        self.maxPool = nn.MaxPool2d(2)

        self.softMax = nn.LogSoftmax()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.output_mask = []

    def forward(self, input):


        conv1_output = self.conv1(input)
        relu1 = self.Relu(conv1_output)
        pool1 = self.maxPool(relu1)


        conv2_output =  self.conv2(relu1)
        relu2 = self.Relu(conv2_output)
        pool2 = self.maxPool(relu2)

        flattened = pool2.view(1, -1)
        linear1 = self.linear1(flattened)
        linear2 = self.linear2(linear1)

        softmax = self.softMax(linear2)

        return softmax

