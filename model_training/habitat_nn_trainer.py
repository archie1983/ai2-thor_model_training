import torch
from torch import nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import HabitatNeuralNetwork, HabitatDataLoading
from time import time

##
# This is where we define loss function, loss rate, decide on which architecture we want,
# and then we train.
##
class HabitatNNTrainer():
    def __init__(self, hp, load_saved = False, pth_path = ''):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hp = hp # hyper params

        # Create the required architecture
        self.model = HabitatNeuralNetwork(hp)

        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            ## If we want to use all GPUs there are
            if hp.USE_PARALLEL_GPUS:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)  # Wrap the model with DataParallel

            ## If we want to use distributed sampler
            if hp.USE_DISTRIBUTED_SAMPLER:
                # Initialize the distributed environment
                dist.init_process_group(backend='nccl')

            # make sure model is on the GPU
            self.model = self.model.to(self.device)

            if hp.USE_DISTRIBUTED_SAMPLER:
                # Wrap the model with DDP
                self.model = DDP(self.model, device_ids=[self.device])

        # Load the data and split it in batches and training and test portions
        dl = HabitatDataLoading(hp)
        (train_data_loader, test_data_loader) = dl.get_train_test_loaders()
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        # Loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # epoch and loss
        self.current_epoch = 0
        self.current_loss = 0

        # if we want to load saved checkpoint, then we will now overwrite hyperparams along with
        # model, optimizer, current loss and current epoch
        if load_saved:
            (self.model, self.optimizer) = self.load_model(self.model, self.optimizer, pth_path)

        ## Print the model for debug purposes
        print(self.model)

    ##
    # This is how we train the network - forward pass, loss calculation, parameter optimization
    # and backward propagation.
    ##
    def train(self):
        size = len(self.train_data_loader.dataset) # how many images we have in total that we'll go through
        batch_counter = 0
        start_time = time() # we'll want to know how fast we are

        # Iterate through the DataLoader
        for images, actions, path_lengths in self.train_data_loader:
            # print(images, actions, path_lengths)
            #print(actions, path_lengths)
            # put out data on the GPU
            images = images.to(self.device)
            actions = actions.to(self.device)
            #path_lengths = path_lengths.to(self.device)

            # make prediction on the current data with the current neural net
            pred = self.model(images)

            # calculate how far off are we
            loss = self.loss_function(pred, actions)

            # back-propagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Every so many batches, tell me what is the current loss
            if (batch_counter % 100) == 0:
                # print(i)
                # what is the latest loss and how many images have we processed
                loss, current = loss.item(), batch_counter * len(images)
                # print(str(i), str(batch))
                time_spent = time() - start_time
                # print it pretty
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], t= {time_spent:>0.1f}")
                self.current_loss = loss

            # update batch counter
            batch_counter += 1

    ##
    # This is how we test the network - forward pass and scoring
    ##
    def test(self):
        # how many images we have in total that we'll go through
        size = len(self.test_data_loader.dataset)
        # how many batches we have of those images
        num_batches = len(self.test_data_loader)
        # initialisation
        test_loss, correct = 0, 0

        with torch.no_grad():
            # take one batch at a time
            batch_counter = 0

            # Iterate through the DataLoader of test data
            for images, actions, path_lengths in self.test_data_loader:
                # make sure our data is on GPU
                images = images.to(self.device)
                actions = actions.to(self.device)

                # make a prediction using current model with current weights and biases
                pred = self.model(images)

                # compare the predictions in this batch with the correct answers
                # and calculate loss from that comparison. The prediction may well
                # be correct, but the logit value (the probability) may be not very
                # much higher than other logits- that will also be taken into account
                # here when calculating loss.
                test_loss += self.loss_function(pred, actions).item()

                # if the index of the highest probability in logits is the same as
                # the expected answer, then count that as a correct answer.
                correct += (pred.argmax(1) == actions).type(torch.float).sum().item()

            # what's the mean loss per batch?
            test_loss /= num_batches
            # what's the mean correctness per image
            correct /= size
            # print it pretty
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            self.current_loss = test_loss

    ##
    # Function for saving a model at any given time. Not just model- the optimizer
    # state too and even number of completed epochs and current loss
    ##
    def save_model(self, model, optimizer, save_path):
        # Check if using DDP/DataParallel
        if self.hp.USE_PARALLEL_GPUS or self.hp.USE_DISTRIBUTED_SAMPLER:
            model_state = model.module.state_dict()  # Unwrap DDP/DataParallel
        else:
            model_state = model.state_dict()  # Single GPU

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': self.current_loss,
            'hyperparams': self.hp
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}!")

    ##
    # Function to load previously saved model, optimizer, loss, epoch and hyperparams
    ##
    def load_model(self, model, optimizer, load_path):
        checkpoint = torch.load(load_path)
        self.hp = checkpoint['hyperparams']

        # Load model state (handle DDP/DataParallel if needed)
        if self.hp.USE_PARALLEL_GPUS or self.hp.USE_DISTRIBUTED_SAMPLER:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and training state (optional)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Update hyperparameters
        self.current_epoch = checkpoint['epoch']
        self.current_loss = checkpoint['loss']

        # We loaded the epoch counter that was correct when the model was saved.
        # But now we will be starting a new epoch and it must be +1.
        self.current_epoch += 1

        print("Loaded ", load_path, " current loss: ", self.current_loss, " current epoch: ", self.current_epoch)

        return model, optimizer

    ##
    # Run all the epochs that we want. One epoch is when we go through all the batches.
    ##
    def do_epochs(self, epochs=50):
        # if state was saved before and we want to start from some epoch
        if self.current_epoch > 0:
            from_step = self.current_epoch
        else:
            from_step = 0

        for t in range(from_step, epochs):
            epoch_start_time = time()
            print(f"Epoch {t + 1}\n-------------------------------")
            self.current_epoch = t
            self.train()
            self.test()
            # after each epoch save the model. This could be improved, i.e. only save model if it's better than last one.
            self.save_model(self.model, self.optimizer, "epoch_" + str(self.current_epoch) + ".pth")
            epoch_run_time = time() - epoch_start_time
            print(f"Epoch ran for: {epoch_run_time:>0.1f} s")
        print("Done!")
