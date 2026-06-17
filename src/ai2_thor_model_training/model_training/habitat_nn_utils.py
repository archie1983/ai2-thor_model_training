##
# This is a collection of utility functions for my Habitat NN package - e.g., it will contain functionality that is
# common between training and inference and may contain other functions too.
##
import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import HabitatNeuralNetwork
from torchvision import transforms

##
# Loads a model architecture as specified in hyper params. Also puts the created architecture
# into the GPU memory or RAM - also depending on hyper params.
#
# Returns an already allocated model.
##
def load_model_architecture(hp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the required architecture
    model = HabitatNeuralNetwork(hp)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        ## If we want to use all GPUs there are
        if hp.USE_PARALLEL_GPUS:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)  # Wrap the model with DataParallel

        ## If we want to use distributed sampler
        if hp.USE_DISTRIBUTED_SAMPLER:
            # Initialize the distributed environment
            dist.init_process_group(backend='nccl')

        # make sure model is on the GPU
        model = model.to(device)

        if hp.USE_DISTRIBUTED_SAMPLER:
            # Wrap the model with DDP
            model = DDP(model, device_ids=[device])
    else:
        # make sure model is on the GPU
        model = model.to(device)

    return model


# Define transforms
habitat_pics_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
