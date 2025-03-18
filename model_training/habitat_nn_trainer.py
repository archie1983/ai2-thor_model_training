import torch
from torch import nn
from . import HabitatNeuralNetwork, HabitatDataLoading

##
# This is where we define loss function, loss rate, decide on which architecture we want,
# and then we train.
##
class HabitatNNTrainer():
    def __init__(self, hp):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hp = hp # hyper params

        # Load the data and split it in batches and training and test portions
        dl = HabitatDataLoading(hp)
        (train_data_loader, test_data_loader) = dl.get_train_test_loaders()
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        # Create the required architecture
        self.model = HabitatNeuralNetwork(hp)

        # Check if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)  # Wrap the model with DataParallel
        self.model = self.model.to('cuda:0')

        # Loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # make sure model is on the GPU
        self.model.to(self.device)
        ## Print the model for debug purposes
        print(self.model)

    ##
    # This is how we train the network - forward pass, loss calculation, parameter optimization
    # and backward propagation.
    ##
    def train(self):
        size = len(self.train_data_loader.dataset) # how many images we have in total that we'll go through
        batch_counter = 0

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
            if (batch_counter % 20) == 0:
                # print(i)
                # what is the latest loss and how many images have we processed
                loss, current = loss.item(), batch_counter * len(images)
                # print(str(i), str(batch))
                # print it pretty
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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

    ##
    # Run all the epochs that we want. One epoch is when we go through all the batches.
    ##
    def do_epochs(self, epochs=50):
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")