import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HabitatDataset(Dataset):
    def __init__(self, pickle_files, hp, image_dir, transform=None):
        """
        Args:
            pickle_files (list): List of paths to pickle files containing the data.
            image_dir (str): Directory where all the images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.pickle_files = pickle_files
        self.image_dir = image_dir
        self.transform = transform

        self.hp = hp # hyper params

        # Load all data from pickle files
        self.data = []
        # take one pickle at a time
        for pickle_file in pickle_files:
            # turning pickle file name, e.g. hm_10.pkl into directory name, e.g. h_10
            habitat_dir = pickle_file.split("/")[-1][:-4].replace("m", "")
            with open(pickle_file, 'rb') as f:
                hab_data = pickle.load(f) # all habitat data in this pickle
                for (expl_length, expl_steps) in hab_data: # go through it and disect each tuple (exploration)

                    # If exploration length doesn't satisfy us, we can skip it
                    if expl_length < hp.min_expl_size:
                        continue
                    #print(expl_length)
                    # Go through all exploration steps and append it to the dataset
                    for step in expl_steps:
                        (action, path_length, img_uris) = step
                        img_uris = [habitat_dir + "/" + iu for iu in img_uris] # img_uris need to be adjusted with the habitat folder
                        #self.data.extend(pickle.load(f))
                        self.data.append((action, path_length, img_uris))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        action, path_length, image_paths = self.data[idx]
        #print("AE::::::", action, path_length, image_paths)

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

        # Convert action to a tensor (you may need to map actions to integers)
        action_tensor = torch.tensor(self._action_to_index(action), dtype=torch.long)

        # Convert path length to a tensor
        path_length_tensor = torch.tensor(path_length, dtype=torch.float32)

        return images, action_tensor, path_length_tensor

    def _action_to_index(self, action):
        """
        Helper function to convert action strings to indices.
        You can customize this based on your specific actions.
        """
        action_mapping = {
            'RotateLeft': 0,
            'RotateRight': 1,
            'MoveAhead': 2,
            'STOP': 3,
            # Add more actions as needed
        }
        return action_mapping.get(action, -1)  # Return -1 for unknown actions