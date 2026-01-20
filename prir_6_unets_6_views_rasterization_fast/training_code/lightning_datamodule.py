#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Lightning Datamodule


# In[8]:


from torch.utils.data import DataLoader, Dataset, random_split
from unetxd import UNet2d
import torch
from torch import nn
import pytorch_lightning as pl
from dataset_generator import *

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, view, recompute_normals, img_width = 800 , img_height = 800, train_val_split=0.8, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.view = view
        self.recompute = recompute_normals
        self.img_width = img_width
        self.img_height = img_height

    def setup(self, stage=None):
        # Split the data into training and validation sets.
        dataset = CustomDataset(datasets_directory = self.data_dir, img_width = self.img_width , img_height = self.img_height, 
                                recompute_normals = self.recompute, view = self.view)
        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        
        print('Total dataset size: ', len(dataset))
        print('Training dataset size: ', train_size, f'({(train_size / len(dataset)) * 100:.2f}%)')
        print('Validation dataset size: ', val_size, f'({(val_size / len(dataset)) * 100:.2f}%)')
        print('Batch size: ', self.batch_size)
        print('Number of (train+val) steps per epoch: ', len(dataset) // self.batch_size)

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive
            pin_memory=True,  # Enable for A6000 GPU-CPU transfer speed
            shuffle=True,
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch more batches
            multiprocessing_context='spawn' if self.num_workers > 0 else None  # Spawn for compatibility
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive
            pin_memory=True,  # Enable for A6000 GPU-CPU transfer speed
            shuffle=False,
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch more batches
            multiprocessing_context='spawn' if self.num_workers > 0 else None  # Spawn for compatibility
        )

    # TO DO: Test DataLoader

'''
# Example usage:
input_path = "/autofs/vast/lemon/temp_stuff/nicolas/aligned_closed_meshes"

data_module = CustomDataModule(data_dir=input_path, batch_size=5, 
                               view ='Top', img_width = 800 , img_height = 800,
                               train_val_split=0.8, num_workers=4)
data_module.setup()
'''

