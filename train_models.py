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
                     learning_rate = 0.001)

#hnt = HabitatNNTrainer(hp3)
hnt = HabitatNNTrainer(hp3, load_saved = True, pth_path = 'accuracy_093.pth')
hnt.do_epochs(150)
