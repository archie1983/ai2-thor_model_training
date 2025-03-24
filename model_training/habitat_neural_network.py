from torch import nn
import torch

##
# Here we will be defining the architecture of our NN model.
# We can pass a parameter to the constructor to determine which
# architecture we want.
##
class HabitatNeuralNetwork(nn.Module):
    def __init__(self, hp):
        super(HabitatNeuralNetwork, self).__init__()
        self.hp = hp
        self.neural_network = self.get_neural_network(hp.architecture_id)

    ##
    # Now the rest of the network- the architecture to achieve nice gradient descent
    ##
    def get_neural_network(self, arch_id):
        if (arch_id == 0):
            lrs = nn.Sequential(
                nn.Flatten(),
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
                nn.Flatten(),
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
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding='same', dilation=1, groups=1,
                          bias=True, padding_mode='zeros'),
                nn.MaxPool2d(5, stride=1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='same', dilation=1,
                          groups=1, bias=True, padding_mode='zeros'),
                nn.MaxPool2d(5, stride=1),
#                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding='same', dilation=1,
#                          groups=1, bias=True, padding_mode='zeros'),
#                nn.MaxPool2d(5, stride=1),
                nn.Dropout(p=0.2),
                nn.Flatten(),
                nn.Linear(746496, 512),
                nn.Sigmoid(),
                nn.Linear(512, 4),
            )

        return lrs

    ##
    # Forward pass. We get the input data x and return the final layer
    # logits values.
    ##
    def forward(self, x):
        logits = self.neural_network(x)
        return logits
