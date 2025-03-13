from model_training import HabitatNNTrainer
hnt = HabitatNNTrainer(architecture_id = 1, batch_size = 10, data_split = [0.9, 0.1], min_expl_size = 0)
hnt.do_epochs(5)
