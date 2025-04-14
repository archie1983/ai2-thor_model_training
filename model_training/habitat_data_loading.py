from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from . import HabitatDataset, habitat_pics_transform
import glob, os, json, torch
import numpy as np

class HabitatDataLoading():
    def __init__(self, hp):
        # Set seed with fallback to 21011983 if not specified
        if not hasattr(hp, 'seed'):
            setattr(hp, 'seed', 21011983)
        # Set random seeds for reproducibility
        self._set_seeds(hp.seed)

        # Define transforms
        self.transform = habitat_pics_transform

        # Create dataset
        h_pkl_files = glob.glob("harvested_data/*.pkl")
        image_dir = 'harvested_data'
        full_dataset = HabitatDataset(h_pkl_files, hp, image_dir, transform=self.transform)

        # Create consistent train/test splits
        train_indices, test_indices = self._get_consistent_splits(full_dataset, hp)

        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)

        ## If we want to use distributed sampler, then DataLoader will require a sampler
        if hp.USE_DISTRIBUTED_SAMPLER:
            # Use a DistributedSampler to split the data
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=hp.seed)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)  # No shuffling for test

            self.train_data_loader = DataLoader(
                train_dataset,
                batch_size=hp.batch_size,
                sampler=train_sampler,
                shuffle=False  # Sampler handles shuffling
            )
            self.test_data_loader = DataLoader(
                test_dataset,
                batch_size=hp.batch_size,
                sampler=test_sampler,
                shuffle=False
            )
        else:
            # Use fixed generator for consistent shuffling
            g = torch.Generator()
            g.manual_seed(hp.seed)

            self.train_data_loader = DataLoader(
                train_dataset,
                batch_size=hp.batch_size,
                shuffle=True,
                generator=g
            )
            self.test_data_loader = DataLoader(
                test_dataset,
                batch_size=hp.batch_size,
                shuffle=False  # Never shuffle test set
            )

    def _set_seeds(self, seed):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_consistent_splits(self, dataset, hp):
        """Get consistent train/test splits across runs"""
        split_file = f"data_splits_seed{hp.seed}.json"

        if os.path.exists(split_file):
            # Load existing splits
            with open(split_file, 'r') as f:
                indices = json.load(f)
            return indices['train'], indices['test']
        else:
            # Create new splits and save them
            train_size = int(len(dataset) * hp.data_split[0])
            test_size = len(dataset) - train_size

            # Use numpy for consistent shuffling
            indices = np.random.permutation(len(dataset)).tolist()
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # Save splits for future runs
            with open(split_file, 'w') as f:
                json.dump({'train': train_indices, 'test': test_indices}, f)

            return train_indices, test_indices

    def get_train_test_loaders(self):
        return self.train_data_loader, self.test_data_loader