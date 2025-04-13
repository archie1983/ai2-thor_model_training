import torch
from model_training import HabitatNNTrainer, HabitatDataLoading
from PIL import Image
import os

##
# This class will analyze provided scene image using a neural network specified in pth_path.
# This will allow querying things like next best action, remaining path length, next dreamed scene given
# the current scene and some action, etc..
##
class SceneAnalyzer():
    def __init__(self, pth_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # First of all load the checkpoint and read hyper params to know how to load the model
        checkpoint = torch.load(pth_path)
        self.hp = checkpoint['hyperparams']

        # Now re-use habitat nn trainer to load the actual model weights
        self.hnt = HabitatNNTrainer(self.hp, True, pth_path)

        # we want a loaded model from the nn trainer
        self.model = self.hnt.model

    ##
    # This function will allow to infer the next best move given an image
    ##
    def next_best_move(self, scene_img_url):
        # When we have an image URL, we want to load it and transform according to how we trained the model
        hdl = HabitatDataLoading()
        self.transform = hdl.transform



    ##
    # Turn image (or images) into a tensor that we can use in Pytorch
    ##
    def image_files_to_tensor(self, image_paths):
        images = []
        # If we only want to use front view, then we will not be stacking all images, but using just one
        if self.hp.use_front_view_only:
            img = Image.open(os.path.join(self.image_dir, image_paths[0])).convert('RGB') # only take one image
            if self.transform:
                img = self.transform(img)
            images = img
        else:
            # Load images
            for img_path in image_paths: # tale all images
                img = Image.open(os.path.join(self.image_dir, img_path)).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)

            # Stack images into a single tensor (assuming all images are the same size)
            images = torch.stack(images)