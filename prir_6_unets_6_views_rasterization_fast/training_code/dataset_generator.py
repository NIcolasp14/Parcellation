#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dataset Generator


# In[7]:


import torch
import numpy as np
from torch.utils.data import Dataset
from utils_wm import *

class CustomDataset(Dataset):
    def __init__(self, datasets_directory, view, recompute_normals, img_width = 800 , img_height = 800):
        self.img_width = img_width
        self.img_height = img_height
        self.datasets_directory = datasets_directory
        self.names, self.m_paths, self.l_paths = validate_files(self.datasets_directory)
        self.view = view
        self.recompute = recompute_normals
        
    def __len__(self):
        return len(self.m_paths)  # Return the length of one of the paths (assuming they have the same length)

    def __getitem__(self, idx):
        m_path = self.m_paths[idx]
        l_path = self.l_paths[idx]
        name = self.names[idx]

        labels = get_labels(l_path)
        # CRITICAL FIX: Disable mesh augmentation for GPU raycasting!
        # With GPU raycasting, mesh caching is essential for speed.
        # Augmentation breaks caching, causing 10x slowdown (30s vs 2s per batch)
        mesh = create_mesh(m_path, perturb_vertices=False, std_dev=0.1)  # perturb_vertices=False for GPU caching!
        extmat = compute_extmat(mesh)
        intmat = compute_intmat(self.img_width, self.img_height)
        rotation_matrices = compute_rotations(random_degs=7, view = self.view)
        extmat = np.matmul(extmat, rotation_matrices)

        output_maps, labels_maps = generate_maps(mesh=mesh, labels=labels,
                                                 intmat=intmat, extmat=extmat,
                                                 img_width=self.img_width, 
                                                 img_height=self.img_height, rotation_matrices=rotation_matrices, recompute_normals = self.recompute) # Will check the shape of exmat and decide if it computes
                                                                             # all maps or just one depending on the specified sel.view
                                                                             # (depends on np.matmul(extmat, rotation_matrices).shape)
        # Get output maps and labels maps as tensors
        output_maps = np.transpose(output_maps[0], (2,0,1))
        output_maps = torch.from_numpy(output_maps)
        labels_maps =  torch.from_numpy(labels_maps[0]).to(torch.int64)
        
        # Convert labels to one-hot encoding
        num_classes = 37  # Number of possible classes (36 + 1 since vaues will go from 0 to 36)
        labels_maps = torch.add(labels_maps,1) # Add 1 since the class -1 is not supported by the one-hot encoding
        labels_maps = torch.nn.functional.one_hot(labels_maps, num_classes=num_classes)
        labels_maps = labels_maps.permute(2, 0, 1).contiguous()
        labels_maps = labels_maps.to(torch.float32)
        
        return output_maps, labels_maps
        
'''                
# Example usage
img_width = 800
img_height = 800
input_path = "/autofs/vast/lemon/temp_stuff/nicolas/aligned_closed_meshes"
view='Top'
custom_dataset = CustomDataset(input_path, view, img_width, img_height)
'''

