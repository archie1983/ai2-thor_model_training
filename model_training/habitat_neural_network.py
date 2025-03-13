from torch import nn
import torch

##
# Here we will be defining the architecture of our NN model.
# We can pass a parameter to the constructor to determine which
# architecture we want.
##
class HabitatNeuralNetwork(nn.Module):
    def __init__(self, architecture=0):
        super(HabitatNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # we'll need this to run data through before we can pass it to linear relu stack
        self.linear_relu_stack = self.get_linear_relu_stack(architecture)
        self.architecture = architecture

    ##
    # Now the rest of the network- the architecture to achieve nice gradient descent
    ##
    def get_linear_relu_stack(self, arch_id):
        if (arch_id == 0):
            lrs = nn.Sequential(
                nn.Linear(24 * 120, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 24),
            )
        elif (arch_id == 1):
            lrs = nn.Sequential(
                nn.Linear(224 * 224 * 3 * 3, 1013), # 224x224 pixel size of image, 3 colour channels (RGB), 3 images stacked
                #nn.Sigmoid(),
                #nn.Linear(2016, 2016),
                #nn.Linear(2016, 1013),
                #nn.Sigmoid(),
                #nn.Linear(2016, 1013),
                nn.Sigmoid(),
                nn.Linear(1013, 512),
                nn.Sigmoid(),
                nn.Linear(512, 512),
                nn.Sigmoid(),
                nn.Linear(512, 4),
            )
        elif (arch_id == 2):
            lrs = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding='same', dilation=1, groups=1,
                          bias=True, padding_mode='zeros'),
                nn.MaxPool2d(5, stride=1),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same', dilation=1,
                          groups=1, bias=True, padding_mode='zeros'),
                nn.MaxPool2d(5, stride=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same', dilation=1,
                          groups=1, bias=True, padding_mode='zeros'),
                nn.MaxPool2d(5, stride=1),
                nn.Dropout(p=0.2),
                nn.Flatten(),
                nn.Linear(82944, 512),
                nn.Sigmoid(),
                nn.Linear(512, 24),
            )

        return lrs

    ##
    # Forward pass. We get the input data x and return the final layer
    # logits values.
    ##
    def forward(self, x):
        x = self.flatten(x)  # all architectures will start with flattening layer
        logits = self.linear_relu_stack(x)
        return logits