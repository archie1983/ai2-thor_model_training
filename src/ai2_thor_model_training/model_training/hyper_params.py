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
    def __init__(self, architecture_id = 1,
                 batch_size = 10,
                 data_split = [0.9, 0.1],
                 min_expl_size = 0,
                 use_front_view_only = False,
                 learning_rate = 0.001,
                 momentum = 0.95,
                 nesterov_momentum = True):
        self.architecture_id = architecture_id
        self.batch_size = batch_size
        self.data_split = data_split
        self.min_expl_size = min_expl_size
        self.use_front_view_only = use_front_view_only
        self.learning_rate = learning_rate # learning rate
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum
        
        # Below are parameters that we keep settable only here and not withing code. This can change, but I thought that it's unlikely
        # we would want to often change these parameters.
        self.USE_DISTRIBUTED_SAMPLER = False
        self.USE_PARALLEL_GPUS = False

        # the same constant random seed to make sure that DataLoader splits data between test and train
        # datasets the same way every time. If it's 0, then we don't use this feature.
        # Whether we use the seed or not, the train/test data split indexes will be store and re-used
        # between runs anyway. This happens in HabitatDataLoading._get_consistent_splits()
        self.seed = 0 # 21011983