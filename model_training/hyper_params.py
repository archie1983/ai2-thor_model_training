##
# This will be a class keeping all hyperparameters in one place. We can then instantiate
# it, set the parameters as needed and pass the instance as a paremeter to the training function.
# We can even store the params in a pickle file.
##
class HyperParameters():
    ##
    # architecture_id : which architecture we would like to use as defined in HabitatNeuralNetwork class
    # batch_size : size of batch (number of training pairs (images :: metrics) in one batch)
    # data_split : data split between training and testing
    # min_expl_size : If exploration contains fewer than this many steps, then we skip it
    # use_front_view_only : Whether we want to use the front view only (True) or also the side images (False)
    ##
    def __init__(self, architecture_id = 1, batch_size = 10, data_split = [0.9, 0.1], min_expl_size = 0, use_front_view_only = False):
        self.architecture_id = architecture_id
        self.batch_size = batch_size
        self.data_split = data_split
        self.min_expl_size = min_expl_size
        self.use_front_view_only = use_front_view_only
        self.USE_DISTRIBUTED_SAMPLER = False

