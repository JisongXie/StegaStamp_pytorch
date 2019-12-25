import os
import utils
import numpy as numpy

import torch
from torch import nn

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if activation is not None:
            if activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return output


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size-1)/2))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class StegaStampEncoder(nn.Module):
    def __init__(self):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(128, 2, activation='relu')
        self.conv6 = Conv2D(128, 3, activation='relu')
        self.up7 = Conv2D(64, 2, activation='relu')
        self.conv7 = Conv2D(64, 3, activation='relu')
        self.up8 = Conv2D(32, 2, activation='relu')
        self.conv8 = Conv2D(32, 3, activation='relu')
        self.up9 = Conv2D(32, 2, activation='relu')
        self.conv9 = Conv2D(32, 3, activation='relu')
        self.residual = Conv2D(3, 1, activation=None)
    
    def call(self, inputs):
        secrect, image = inputs
        secrect = secrect - .5
        image = image - .5

        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(50, 50, 3)
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)

        inputs = torch.cat([secrect_enlarged, image], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))
        merge8 = torch.cat([conv2,up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self):
        self.decoder = nn.Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), strides=2, activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), strides=2, activation='relu'),
            Conv2D(128, (3, 3), strides=2, activation='relu'),
            Conv2D(128, (3, 3), strides=2, activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(secret_size)
        ])

    def forward(self, image):
        image = image - .5
        return self.decoder(image)

