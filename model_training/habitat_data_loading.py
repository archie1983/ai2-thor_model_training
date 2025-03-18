from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from . import HabitatDataset
import glob

class HabitatDataLoading():
    def __init__(self, hp):
        # Define the transform for preprocessing images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # List of pickle files and image directory
        h_pkl_files = glob.glob("harvested_data/*.pkl")
        image_dir = 'harvested_data'

        # Create the dataset
        dataset = HabitatDataset(h_pkl_files, hp, image_dir, transform=self.transform)

        if hp.USE_DISTRIBUTED_SAMPLER:
            # Use a DistributedSampler to split the data
            train_sampler = DistributedSampler(dataset)

        train_dataset, test_dataset = random_split(dataset, hp.data_split)

        if hp.USE_DISTRIBUTED_SAMPLER:
            self.train_data_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, sampler=train_sampler)
            self.test_data_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True, sampler=train_sampler)
        else:
            self.train_data_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
            self.test_data_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=True)

    def get_train_test_loaders(self):
        return (self.train_data_loader, self.test_data_loader)