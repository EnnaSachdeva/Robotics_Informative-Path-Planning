import torch
import torch.nn as nn
from networkFolder.net import PConv2d



class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = PConv2d(1, 64, 3, 2, 3)
        self.conv2 = PConv2d(64, 128, 3, 2, 1)
        self.conv3 = PConv2d(128, 256, 3, 2, 1)
        self.conv4 = PConv2d(256, 512, 3, 2, 1)
        #self.conv5 = PConv2d(512, 512, 3, 2, 1)
        #self.conv6 = PConv2d(512, 512, 3, 2, 1)
        #self.conv7 = PConv2d(512, 512, 3, 2, 1)
        #self.conv8 = PConv2d(512, 512, 3, 2, 1)

        self.conv8_1 = PConv2d(384, 128, 3, 1, 1)
        self.conv8_2 = PConv2d(192, 64, 3, 1, 0)
        self.conv9 = PConv2d(65, 1, 3, 1, 1)
        #self.conv10 = PConv2d(384, 128, 3, 1, 1)
        #self.conv11 = PConv2d(192, 64, 3, 1, 1)
        #self.conv12 = PConv2d(68, 1, 3, 1, 1)

        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.batchNorm4 = nn.BatchNorm2d(512)
        self.batchNorm8_1 = nn.BatchNorm2d(128)
        self.batchNorm9 = nn.BatchNorm2d(256)
        self.batchNorm10 = nn.BatchNorm2d(128)
        self.batchNorm11 = nn.BatchNorm2d(64)

        self.Relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.output_mask = []

    def forward(self, input, input_mask):


        conv1_output, conv1_output_mask = self.conv1(input, input_mask)
        batchNorm1_output = self.batchNorm1(conv1_output)
        # print(batchNorm1_output.shape)
        relu1_output =  self.Relu(batchNorm1_output)
        # 256x256
        conv2_output, conv2_output_mask =  self.conv2(relu1_output, conv1_output_mask)
        batchNorm2_output =  self.batchNorm2(conv2_output)
        # print(conv2_output.shape)
        relu2_output =  self.Relu(batchNorm2_output)
        # 128x128
        conv3_output, conv3_output_mask =  self.conv3(relu2_output, conv2_output_mask)
        batchNorm3_output =  self.batchNorm3(conv3_output)
        relu3_output =  self.Relu(batchNorm3_output)


        upsample3 =  self.upsample(relu3_output)
        upsample3_mask =  self.upsample(conv3_output_mask)
        concat3 = torch.cat((upsample3, relu2_output), 1)
        concat3_mask = torch.cat((upsample3_mask, conv2_output_mask), 1)
        conv11_output, conv11_output_mask =  self.conv8_1(concat3, concat3_mask)
        batchNorm11_output =  self.batchNorm8_1(conv11_output)
        leakyRelu3 =  self.leakyRelu(batchNorm11_output)


        upsample4 =  self.upsample(leakyRelu3)
        upsample4_mask =  self.upsample(conv11_output_mask)
        concat4 = torch.cat((upsample4, conv1_output), 1)
        concat4_mask = torch.cat((upsample4_mask, conv1_output_mask), 1)
        conv12_output, conv12_output_mask =  self.conv8_2(concat4, concat4_mask)
        batchNorm12_output =  self.batchNorm11(conv12_output)
        leakyRelu4 =  self.leakyRelu(batchNorm12_output)


        upsample5 =  self.upsample(leakyRelu4)
        upsample5_mask =  self.upsample(conv12_output_mask)
        concat5 = torch.cat((upsample5, input), 1)
        concat5_mask = torch.cat((upsample5_mask, input_mask), 1)
        conv13_output, conv13_output_mask =  self.conv9(concat5, concat5_mask)

        leakyRelu5 =  self.leakyRelu(conv13_output)

        self.output_mask = conv13_output_mask

        return leakyRelu5

