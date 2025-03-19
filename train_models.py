from model_training import HabitatNNTrainer, HyperParameters

hp1 = HyperParameters(architecture_id = 1,
                     batch_size = 10,
                     data_split = [0.9, 0.1],
                     min_expl_size = 0,
                     use_front_view_only = False) # mickey mouse and doesn't converge, but does train

hp2 = HyperParameters(architecture_id = 2,
                     batch_size = 10,
                     data_split = [0.9, 0.1],
                     min_expl_size = 0,
                     use_front_view_only = True)

hnt = HabitatNNTrainer(hp2)
#hnt = HabitatNNTrainer(hp, load_saved = True, pth_path = 'checkpoint.pth')
hnt.do_epochs(5)
