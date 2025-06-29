import torch
from numpy.ma.core import argmax
from torch import softmax

from model_training import load_model_architecture, habitat_pics_transform, index_to_action
from PIL import Image

##
# This class will analyze provided scene image using a neural network specified in pth_path.
# This will allow querying things like next best action, remaining path length, next dreamed scene given
# the current scene and some action, etc..
##
class SceneAnalyzer():
    def __init__(self, pth_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("AAAAAAAAAAAAAAAAAAAAAAA: ", self.device)

        # When we have an image URI, we want to load it and transform according to how we trained the model
        self.transform = habitat_pics_transform

        # First of all load the checkpoint and read hyper params to know how to load the model
        checkpoint = torch.load(pth_path, map_location=torch.device(self.device))
        self.hp = checkpoint['hyperparams']

        self.model = load_model_architecture(self.hp)

        ## Print the model for debug purposes
        print(self.model)

        # Load model state (handle DDP/DataParallel if needed)
        if self.hp.USE_PARALLEL_GPUS or self.hp.USE_DISTRIBUTED_SAMPLER:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()

    ##
    # This function will allow to infer the next best move given an image
    ##
    def next_best_move(self, scene_img_url = "", raw_img = None):
        if raw_img == None and len(scene_img_url) > 0:
            img_tensor = self.image_files_to_tensor([scene_img_url])
        elif(raw_img != None and len(scene_img_url) == 0):
            #img = Image.open(image_paths[0]).convert('RGB') # only take one image
            img = raw_img.convert('RGB')
            if self.transform:
                # the .unsqueeze(0) bit adds batch dimensionality to the input, which is required by the net.
                # e.g., the net expects input of [1, 3, 224, 224] geometry and not [3, 224, 224].
                img = self.transform(img).unsqueeze(0)
            img_tensor = img
        else:
            print("Either image URI or a raw image must be provided, but not both and not none.")
            return

        with torch.no_grad():
            # make sure our data is on GPU
            images = img_tensor.to(self.device)
            # make a prediction using current model
            pred = self.model(images)

        print("CNN Pred: ", pred)
        pred = pred.cpu()
        #print("CNN argmax: ", argmax(pred))
        #print(index_to_action(argmax(pred)))
        print("softmax", softmax(pred, dim=1))
        pred = softmax(pred, dim=1)

        return (index_to_action(argmax(pred)), argmax(pred), pred)

    ##
    # Turn image (or images) into a tensor that we can use in Pytorch
    ##
    def image_files_to_tensor(self, image_paths):
        images = []
        # If we only want to use front view, then we will not be stacking all images, but using just one
        if self.hp.use_front_view_only:
            img = Image.open(image_paths[0]).convert('RGB') # only take one image
            if self.transform:
                # the .unsqueeze(0) bit adds batch dimensionality to the input, which is required by the net.
                # e.g., the net expects input of [1, 3, 224, 224] geometry and not [3, 224, 224].
                img = self.transform(img).unsqueeze(0)
            images_tensor = img
        else:
            # Load images
            for img_path in image_paths: # tale all images
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    # the .unsqueeze(0) bit adds batch dimensionality to the input, which is required by the net.
                    # e.g., the net expects input of [1, 3, 224, 224] geometry and not [3, 224, 224].
                    img = self.transform(img).unsqueeze(0)
                images.append(img)

            # Stack images into a single tensor (assuming all images are the same size)
            images_tensor = torch.stack(images)

        return images_tensor

#if __name__ == "__main__":
#    sa = SceneAnalyzer("accuracy_093.pth")
#    sa.next_best_move("harvested_data/h_7/expl_9/3.png")
