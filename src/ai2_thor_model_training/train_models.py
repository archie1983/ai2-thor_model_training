from model_training import HabitatNNTrainer, HyperParameters

hp1 = HyperParameters(architecture_id = 1,
                     batch_size = 10,
                     data_split = [0.9, 0.1],
                     min_expl_size = 0,
                     use_front_view_only = False,
                     learning_rate = 0.001) # mickey mouse and doesn't converge, but does train

hp2 = HyperParameters(architecture_id = 2,
                     batch_size = 10,
                     data_split = [0.9, 0.1],
                     min_expl_size = 0,
                     use_front_view_only = True,
                     learning_rate = 0.001)

hp3 = HyperParameters(architecture_id = 2,
                      batch_size = 10,
                      data_split = [0.8, 0.2],
                      min_expl_size = 0,
                      use_front_view_only = True,
                      learning_rate = 0.001,
                      momentum = 0.95,
                      nesterov_momentum = True)

hp4 = HyperParameters(architecture_id = 2,
                      batch_size = 10,
                      data_split = [0.8, 0.2],
                      min_expl_size = 0,
                      use_front_view_only = True,
                      learning_rate = 0.001,
                      momentum = 0,
                      nesterov_momentum = False)

#hnt = HabitatNNTrainer(hp3)
#hnt = HabitatNNTrainer(hp3, load_saved = True, pth_path = 'accuracy_093.pth')
hnt = HabitatNNTrainer(hp4)
#hnt = HabitatNNTrainer(hp4, load_saved = True, pth_path = 'best_epoch_0.5669423114786375.pth')
#hnt = HabitatNNTrainer(hp4, load_saved = True, pth_path = 'accuracy_068.pth')
hnt.do_epochs(150)
