# Python packages
import os

# Third party packages
import torch
import torchvision.transforms as transforms

# Local packages
from .coco_dataset import CocoDataset
from .cub_dataset import CubDataset
from utils.transform import get_transform
from .a3ds_dataset import A3DSDataset

class DataPreparation:
    def __init__(self, dataset_name='coco', data_path='./data'):
        if dataset_name == 'coco':
            self.DatasetClass = CocoDataset
        elif dataset_name == 'cub':
            self.DatasetClass = CubDataset
        elif dataset_name == '3d':
            self.threed = True
            self.DatasetClass = A3DSDataset
        self.data_path = os.path.join(data_path, self.DatasetClass.dataset_prefix)

    def get_dataset(self, split='train', vision_model=None, vocab=None,
            tokens=None):
        
        ## here the DatasetClass is replaced by the dataset eg : CocoDataset and the init function of CocoDataset is called.
        # Load the dataset 
        dataset = self.DatasetClass()
        self.dataset = dataset
        return self.dataset

    def get_loader(self, dataset, batch_size=128, num_workers=4):
        assert isinstance(dataset, self.DatasetClass)
        # Load the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=dataset.collate_fn)
        return dataloader


    def get_dataset_and_loader(self, split='train', vision_model=None,
            vocab=None, tokens=None, batch_size=2, num_workers=4):
        dataset = self.get_dataset(split, vision_model, vocab, tokens)
        loader = self.get_loader(dataset, batch_size, num_workers)
        return dataset, loader

