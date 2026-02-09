#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Utils


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import nibabel as nib
import cv2
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from scipy.interpolate import RegularGridInterpolator as rgi


# ============================================================================
# GPU ACCELERATION SETUP
# ============================================================================
# Check GPU availability (ENABLED for A6000 48GB GPU)
USE_GPU = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if USE_GPU else 'cpu')

# GPU Raycasting with PyTorch3D (OptiX disabled, PyTorch3D optimized)
USE_GPU_RAYCASTING = False  # CPU benchmark
USE_OPTIX_RAYCASTING = False  
GPU_RAYCASTING_BACKEND = None

# Try optimized PyTorch3D GPU raycasting (with mesh caching)
try:
    if USE_GPU:
        import pytorch3d
        from gpu_raycasting_pytorch3d_fast import generate_maps_gpu  # FAST VERSION (bin_size=None)
        # USE_GPU_RAYCASTING will be set by network_trainer_wm.py based on --raycasting arg
        GPU_RAYCASTING_BACKEND = "PyTorch3D-FAST"
        print("[GPU] PyTorch3D loaded - GPU raycasting AVAILABLE (FAST MODE: bin_size=None)")
except ImportError:
    print("[GPU] PyTorch3D not available - using Open3D CPU raycasting")

# Open3D GPU detection - try to create CUDA device, fallback to CPU
O3D_GPU_AVAILABLE = False
try:
    if USE_GPU:
        # Try to create a tensor on CUDA device to test if it works
        test_device = o3d.core.Device("CUDA:0")
        test_tensor = o3d.core.Tensor([1.0], device=test_device)
        O3D_GPU_AVAILABLE = True
        O3D_DEVICE = test_device
        del test_tensor  # Clean up
    else:
        O3D_DEVICE = o3d.core.Device("CPU:0")
except:
    O3D_GPU_AVAILABLE = False
    O3D_DEVICE = o3d.core.Device("CPU:0")

def print_gpu_status():
    """Print GPU status - call this AFTER raycasting backend is configured"""
    print(f"[GPU] PyTorch CUDA available: {USE_GPU}")
    if USE_GPU:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        print(f"[GPU] Device: {TORCH_DEVICE} ({gpu_name})")
        print(f"[GPU] Augmentations: Gaussian noise, Affine, Elastic (GPU-accelerated)")
        if USE_GPU_RAYCASTING:
            if GPU_RAYCASTING_BACKEND == "OptiX":
                print(f"[GPU] Raycasting: OptiX (GPU) - HARDWARE RT MODE (~0.5-1s per batch)")
            elif GPU_RAYCASTING_BACKEND == "PyTorch3D":
                print(f"[GPU] Raycasting: PyTorch3D (GPU) - FAST MODE (~1-2s per batch)")
        else:
            print(f"[GPU] Raycasting: Open3D (CPU) - FALLBACK MODE (~4s per batch)")
            print(f"[GPU] Install triro for OptiX or PyTorch3D for GPU raycasting speedup")
    else:
        print(f"[GPU] Using CPU for all operations")


# ============================================================================
# GPU-ACCELERATED AUGMENTATION FUNCTIONS
# ============================================================================

def affine_augment_gpu(vertices_torch):
    """
    GPU-accelerated affine augmentation using PyTorch.
    
    Args:
        vertices_torch: torch.Tensor of shape (N, 3) on GPU
    
    Returns:
        torch.Tensor of augmented vertices on GPU
    """
    device = vertices_torch.device
    
    # Generate random rotation angles (-2 to +2 degrees)
    theta_x = torch.deg2rad(torch.FloatTensor(1).uniform_(-2.0, 2.0)).to(device)
    theta_y = torch.deg2rad(torch.FloatTensor(1).uniform_(-2.0, 2.0)).to(device)
    theta_z = torch.deg2rad(torch.FloatTensor(1).uniform_(-2.0, 2.0)).to(device)
    
    # Generate random scaling factors
    scale_x = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
    scale_y = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
    scale_z = torch.FloatTensor(1).uniform_(0.9, 1.1).to(device)
    
    # Generate random shearing factors
    shear_xy = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    shear_xz = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    shear_yx = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    shear_yz = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    shear_zx = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    shear_zy = torch.FloatTensor(1).uniform_(-0.012, 0.012).to(device)
    
    # Create rotation matrices on GPU
    cos_x, sin_x = torch.cos(theta_x), torch.sin(theta_x)
    cos_y, sin_y = torch.cos(theta_y), torch.sin(theta_y)
    cos_z, sin_z = torch.cos(theta_z), torch.sin(theta_z)
    
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos_x, -sin_x],
                       [0, sin_x, cos_x]], device=device, dtype=torch.float32).squeeze()
    
    Ry = torch.tensor([[cos_y, 0, sin_y],
                       [0, 1, 0],
                       [-sin_y, 0, cos_y]], device=device, dtype=torch.float32).squeeze()
    
    Rz = torch.tensor([[cos_z, -sin_z, 0],
                       [sin_z, cos_z, 0],
                       [0, 0, 1]], device=device, dtype=torch.float32).squeeze()
    
    # Create scaling matrix
    S = torch.diag(torch.tensor([scale_x, scale_y, scale_z], device=device, dtype=torch.float32).squeeze())
    
    # Create shearing matrix
    H = torch.tensor([[1, shear_xy, shear_xz],
                      [shear_yx, 1, shear_yz],
                      [shear_zx, shear_zy, 1]], device=device, dtype=torch.float32).squeeze()
    
    # Combine transformations: vertices @ (Rz @ Ry @ Rx @ S @ H)^T
    transform = Rz @ Ry @ Rx @ S @ H
    augmented_vertices = vertices_torch @ transform.T
    
    return augmented_vertices


def elastic_augment_gpu(vertices_torch):
    """
    GPU-accelerated elastic deformation using PyTorch grid_sample.
    
    Args:
        vertices_torch: torch.Tensor of shape (N, 3) on GPU
    
    Returns:
        torch.Tensor of augmented vertices on GPU
    """
    device = vertices_torch.device
    
    # Random grid size
    box_sizes = [2, 3, 4, 5]
    box_size = random.choice(box_sizes)
    
    # Random sigma
    sigma = 5.0 * torch.rand(1, device=device).item()
    
    # Generate random displacement field on GPU
    displacement_field = sigma * torch.randn(1, 3, box_size, box_size, box_size, 
                                             device=device, dtype=torch.float32)
    
    # Normalize vertices to [0, 1]
    mini = vertices_torch.min(dim=0, keepdim=True)[0]
    maxi = vertices_torch.max(dim=0, keepdim=True)[0]
    vertices_norm = (vertices_torch - mini) / (maxi - mini + 1e-8)
    
    # Convert to grid_sample coordinates [-1, 1]
    vertices_grid = vertices_norm * 2.0 - 1.0
    
    # Reshape for grid_sample: (N, 3) -> (1, N, 1, 1, 3)
    vertices_grid = vertices_grid.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    
    # Sample displacements using trilinear interpolation
    # grid_sample expects (N, C, D, H, W) and grid (N, D_out, H_out, W_out, 3)
    displacements = F.grid_sample(displacement_field, vertices_grid, 
                                   mode='bilinear', padding_mode='border', 
                                   align_corners=True)
    
    # Reshape back: (1, 3, N, 1, 1) -> (N, 3)
    displacements = displacements.squeeze(0).squeeze(-1).squeeze(-1).T
    
    # Apply displacements
    elastic_vertices = vertices_torch + displacements
    
    return elastic_vertices


'''
colors_array = annot_data[1][:, :3]
normalized_colors = colors_array / 255.0
updated_colors = np.insert(normalized_colors, 0, [0, 0, 0], axis=0)
cmap_labels = ListedColormap(updated_colors)
'''
def get_labels(annotations_path):
    """
    Obtains Labels associated with vertices of the annotations file

    Parameters:
    - path: Anotations file path

    Returns:
    - labels: list of labels for each vertex
    """
    annot_data = nib.freesurfer.io.read_annot(annotations_path)
    labels = annot_data[0]
    return labels

def create_mesh(mesh_path, perturb_vertices = True, std_dev = 0.1):
    """
    Create a 3D triangle mesh from a FreeSurfer surface file.
    Now with GPU-accelerated augmentations (when not in DataLoader worker)!

    This function reads a FreeSurfer surface file from the specified `mesh_path`,
    processes the vertex and face data, and constructs a 3D triangle mesh.

    Parameters:
    - mesh_path (str): The path to the FreeSurfer surface file to be processed.

    Returns:
    - mesh (o3d.t.geometry.TriangleMesh): A 3D triangle mesh representation of
      the input FreeSurfer surface.

    Dependencies: nibabel (nib), numpy (np), open3d (o3d)
    """
    # Read the FreeSurfer surface file and retrieve vertices, faces, and metadata.
    vertices, faces, info = nib.freesurfer.read_geometry(mesh_path, read_metadata=True)

    # Center the vertices around the origin.
    vertices = vertices - np.mean(vertices, axis=0)

    # Reorder the vertex columns for compatibility with open3d.
    vertices = vertices[:, [2, 0, 1]]

    # If perturb vertices is true, do augmentation
    if perturb_vertices == True:
        
        std_dev *= np.random.rand(1) 
        std_dev_scalar = float(std_dev[0]) if isinstance(std_dev, np.ndarray) else float(std_dev)
        
        # Check if we're in a DataLoader worker (CUDA doesn't work in workers!)
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        use_gpu_here = USE_GPU and (worker_info is None)  # Only use GPU in main process
        
        if use_gpu_here:
            # GPU-ACCELERATED AUGMENTATION PATH (main process only)
            # Convert to PyTorch tensor on GPU
            vertices_torch = torch.from_numpy(vertices).float().to(TORCH_DEVICE)
            
            # Add Gaussian noise on GPU (use scalar std_dev)
            gaussian_noise = torch.randn_like(vertices_torch) * std_dev_scalar
            vertices_torch = vertices_torch + gaussian_noise
            
            # GPU-accelerated affine augmentation
            vertices_torch = affine_augment_gpu(vertices_torch)
            
            # GPU-accelerated elastic augmentation
            vertices_torch = elastic_augment_gpu(vertices_torch)
            
            # Convert back to CPU numpy
            vertices = vertices_torch.cpu().numpy()
        else:
            # CPU PATH (workers or when GPU unavailable)
            gaussian_noise = np.random.normal(0, std_dev, vertices.shape)
            vertices = vertices + gaussian_noise
            vertices = affine_augment(vertices)
            vertices = elastic_augment(vertices)

    # Create Open3D mesh on CPU (raycasting happens in background workers now!)
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.float32(vertices)),
        o3d.core.Tensor(np.int64(faces))
    )
    
    # Compute vertex normals and triangle normals for the mesh.
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    return mesh


def compute_extmat(mesh):
    """
    Compute the external transformation matrix (extmat) for a 3D mesh.

    This function calculates the external transformation matrix `extmat` for
    a 3D mesh, which can be used for various transformations such as centering
    and scaling the mesh.

    Parameters:
    - mesh (o3d.t.geometry.TriangleMesh): The 3D triangle mesh to compute the
      external transformation matrix for.

    Returns:
    - extmat (numpy.ndarray): A 4x4 transformation matrix represented as a
      NumPy array.
    """
    # Calculate the minimum and maximum corners of the mesh's bounding box.
    corner1 = np.min(mesh.vertex.positions.numpy(), axis=0)
    corner2 = np.max(mesh.vertex.positions.numpy(), axis=0)

    # Calculate the midpoint of the bounding box.
    midpoint = (corner1 + corner2) / 2

    # Create an identity 4x4 transformation matrix.
    extmat = np.eye(4)

    # Modify the diagonal elements and the last column of the matrix.
    np.fill_diagonal(extmat, [-1, 1, 1, 1])
    extmat[:,-1] = [-midpoint[0], -midpoint[1], -7.5 * corner1[2], 1]


    return extmat

def compute_intmat(img_width, img_height):
    """
    Compute the intrinsic matrix (intmat) for a camera with given image dimensions.

    Parameters:
    - img_width (int): The width of the camera image in pixels.
    - img_height (int): The height of the camera image in pixels.

    Returns:
    - intmat (numpy.ndarray): A 3x3 intrinsic matrix represented as a NumPy array.
    """
    # Create an identity 3x3 intrinsic matrix
    intmat = np.eye(3)

    # Fill the diagonal elements with appropriate values
    np.fill_diagonal(intmat, [-(img_width + img_height) / 1, -(img_width + img_height) / 1, 1])

    # Set the last column of the matrix for image centering
    intmat[:,-1] = [img_width / 2, img_height / 2, 1]

    return intmat

def compute_rotations(random_degs=5, view = 'Random', random = False):
    """
    Compute six random 3D rotation matrices for Front, Top, Bottom, Left, Back, Right views in this order
    with randomized small rotations from -3 to +3 degrees.

    Returns:
    - rotation_matrices (list of numpy.ndarray): A list containing six 4x4
      rotation matrices represented as NumPy arrays.

    Notes:
    - The rotation matrices are created based on random pitch and yaw angles
      with small random variations.
    """
    
    # Initialize an empty list to store the rotation matrices
    rotation_matrices = []

    if view == 'Random_6':
        # Select a random view from the available options
        available_views = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']
        view = np.random.choice(available_views)
        
    if view == 'All':
        # Define the pitch angles (Front, Bottom, Top) and add random variations
        pitch_angles = [0, 90, 270]
        pitch_angles = np.deg2rad(pitch_angles + np.random.uniform(-random_degs, random_degs, len(pitch_angles)))
        
        # Define the yaw angles (Right, Back, Left) and add random variations
        yaw_angles = [90, 180, 270]
        yaw_angles = np.deg2rad(yaw_angles + np.random.uniform(-random_degs, random_degs, len(yaw_angles)))
        
        # Loop through each pitch angle in radians and create the rotation matrix
        for angle in pitch_angles:
            R_pitch = create_pitch_rotation_matrix(angle)
            R = (R_pitch 
                 @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
                 @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
            
            rotation_matrices.append(R)
            
        # Loop through each yaw angle in radians and create the rotation matrix
        for angle in yaw_angles:
            R_yaw = create_yaw_rotation_matrix(angle)
            R = (R_yaw 
                 @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
                 @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
            rotation_matrices.append(R)

    elif view == 'Front': # Set this to recompute normals on the fly
        angle = np.deg2rad(np.random.uniform(-random_degs, random_degs))
        R = create_pitch_rotation_matrix(angle)
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Bottom':
        angle = np.deg2rad(90 + np.random.uniform(-random_degs, random_degs))
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Top':
        angle = np.deg2rad(270 + np.random.uniform(-random_degs, random_degs))
        R =  (create_pitch_rotation_matrix(angle)
             @ create_yaw_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Right':
        angle = np.deg2rad(90 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle) 
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)

    elif view == 'Back':
        angle = np.deg2rad(180 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle) 
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)
    elif view == 'Left':
        angle = np.deg2rad(270 + np.random.uniform(-random_degs, random_degs))
        R = (create_yaw_rotation_matrix(angle) 
             @ create_pitch_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))) 
             @ create_roll_rotation_matrix(np.deg2rad(np.random.uniform(-random_degs, random_degs))))
        rotation_matrices.append(R)
        
    elif view == 'Random':
        R = (create_yaw_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand()) 
             @ create_pitch_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand()) 
             @ create_roll_rotation_matrix(-np.pi + 2 * np.pi * np.random.rand())) 
        rotation_matrices.append(R)
    rotation_matrices = np.array(rotation_matrices)

    
    #rotation_matrices = np.transpose(rotation_matrices, (1, 2, 0))

    return rotation_matrices



def generate_maps(mesh, labels, intmat, extmat, img_width, img_height, rotation_matrices, recompute_normals):
    """
    Generate the output map based on ray casting and mesh properties.
    views are in this order ALWAYS = ['Front', 'Bottom', 'Top', 'Right', 'Back', 'Left']

    Parameters:
    - mesh (o3d.t.geometry.TriangleMesh): The 3D triangle mesh to cast rays onto.
    - labels (numpy.ndarray): The labels associated with the vertices of the mesh.
    - intmat (numpy.ndarray): A 3x3 intrinsic matrix for camera calibration.
    - extmat (numpy.ndarray): A 4x4 external transformation matrix for camera pose.
    - img_width (int): The width of the camera image in pixels.
    - img_height (int): The height of the camera image in pixels.

    Returns:
    - output_maps(6, 1080, 1920, 3), labels_maps((6, 1080, 1920), ids_maps(6, 1080, 1920), vertex_maps(6, 1080, 1920,3)

    Notes:
    - This function performs ray casting on the provided mesh using the given
      camera parameters and computes an output map based on the cast rays.

    Example:
    >>> mesh = create_mesh("example_mesh.surf")
    >>> labels = get_labels(annotations_path)
    >>> intmat = compute_intmat(1920, 1080)
    >>> extmat = compute_extmat(mesh)
    >>> width = 1920
    >>> height = 1080
    >>> output_map, labels_map = generate_output_map(mesh, intmat, extmat, width, height)
    >>> print(output_map)
    >>> print(labels_map)

    """
    
    # ============================================================================
    # GPU RAYCASTING PATH (OptiX/PyTorch3D) - Hardware RT or GPU accelerated
    # ============================================================================
    if USE_GPU_RAYCASTING:
        return generate_maps_gpu(mesh, labels, intmat, extmat, img_width, img_height, 
                                  rotation_matrices, recompute_normals)
    
    # ============================================================================
    # CPU RAYCASTING PATH (Open3D) - Fallback if GPU raycasting not available
    # ============================================================================

    # Validate parameters using assert statements
    assert isinstance(mesh, o3d.t.geometry.TriangleMesh), "mesh should be of type o3d.t.geometry.TriangleMesh"
    assert isinstance(labels, np.ndarray), "labels should be a 1-D NumPy array"
    expected_shape = (mesh.vertex.normals.shape[0],)
    assert labels.shape == expected_shape, f"labels should have the shape {expected_shape} which is the number of vertices, but got {labels.shape}"    
    assert isinstance(intmat, np.ndarray) and intmat.shape == (3, 3), "intmat should be a 3x3 NumPy array"
    assert isinstance(extmat, np.ndarray) and (extmat.shape == (1, 4, 4) or extmat.shape == (6, 4, 4)), "extmat should be a 4x4 or 6x4x4 NumPy array"
    assert isinstance(img_width, int) and img_width > 0, "img_width should be a positive integer"
    assert isinstance(img_height, int) and img_height > 0, "img_height should be a positive integer"

    # Create a RaycastingScene and add the mesh to it
    # Assuming 'View' argument will never be 'All':
    if recompute_normals == True:
        mesh.vertex.normals = mesh.vertex.normals@np.transpose(rotation_matrices[0][:3,:3].astype(np.float32))
        mesh.triangle.normals = mesh.triangle.normals@np.transpose(rotation_matrices[0][:3,:3].astype(np.float32))
        
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    output_maps = []
    labels_maps = []
    ids_maps = []
    vertex_maps = []

    # rotation_matrices = compute_rotations(random_degs=7, view = view) Given as an argument
    for i in range(rotation_matrices.shape[0]): # TO DO - DONE: ROTATION MATRICES IS NOT DEFINED INSIDE THIS FUNCTION
        # Create rays using pinhole camera model
        rays = scene.create_rays_pinhole(intmat, extmat[i], img_width, img_height)
    
        # Cast rays and retrieve primitive IDs, hit distances, and normals
        cast = scene.cast_rays(rays)
        ids_map = np.array(cast['primitive_ids'].numpy(), dtype=np.int32)
        ids_maps.append(ids_map)
        hit_map = np.array(cast['t_hit'].numpy(), dtype=np.float32)
        weights_map = np.array(cast['primitive_uvs'].numpy(), dtype=np.float32)
        missing_weight = 1 - np.sum(weights_map, axis=2, keepdims=True)
        label_ids = np.argmax(np.concatenate((weights_map, missing_weight), axis=2), axis=2)
        
        # Compute the normal map
        normal_map = np.array(mesh.triangle.normals[ids_map.clip(0)].numpy(), dtype=np.float32)
        normal_map[ids_maps[i] == -1] = [0, 0, -1]
        normal_map[:, :, -1] = -normal_map[:, :, -1].clip(-1, 0)
        normal_map = normal_map * 0.5 + 0.5
    
        # Compute the vertex map
        vertex_map = np.array(mesh.triangle.indices[ids_map.clip(0)].numpy(), dtype=np.int32)
        vertex_map[ids_map == -1] = [-1]
        vertex_maps.append(vertex_map)
    
        # Compute the inverse distance map
        inverse_distance_map = 1 / hit_map
    
        # Compute the coded map with inverse distance
        coded_map_inv = normal_map * inverse_distance_map[:, :, None]
    
        # Normalize the output map
        output_map = (coded_map_inv - np.min(coded_map_inv)) / (np.max(coded_map_inv) - np.min(coded_map_inv))
        output_maps.append(output_map)
    
        # Compute the labels map
        labels_map = labels[vertex_map.clip(0)]
        labels_map[vertex_map == -1] = -1
        #labels_map = np.median(labels_map, axis=2)
        labels_map = labels_map[np.arange(labels_map.shape[0])[:, np.newaxis], np.arange(labels_map.shape[1]), label_ids]
        labels_map = labels_map.astype('float64')
        labels_maps.append(labels_map)

    output_maps = np.array(output_maps)
    labels_maps = np.array(labels_maps)
    #print('Type: ',labels_maps.dtype)
    # ids_maps = np.array(ids_maps)
    # vertex_maps = np.array(vertex_maps)
    
    return output_maps, labels_maps
    # return output_maps, labels_maps, ids_maps, vertex_maps

    

def create_pitch_rotation_matrix(pitch_angle):
    """
    Create a rotation matrix for pitch rotation.

    Parameters:
    - pitch_angle: Angle in radians for pitch rotation.

    Returns:
    - R_pitch: Rotation matrix for pitch.
    """
    R_pitch = np.array([[1, 0, 0, 0],
                        [0, np.cos(pitch_angle), -np.sin(pitch_angle), 0],
                        [0, np.sin(pitch_angle), np.cos(pitch_angle), 0],
                        [0, 0, 0, 1]])
    return R_pitch

def create_yaw_rotation_matrix(yaw_angle):
    """
    Create a rotation matrix for yaw rotation.

    Parameters:
    - yaw_angle: Angle in radians for yaw rotation.

    Returns:
    - R_yaw: Rotation matrix for yaw.
    """
    R_yaw = np.array([[np.cos(yaw_angle), 0, np.sin(yaw_angle), 0],
                      [0, 1, 0, 0],
                      [-np.sin(yaw_angle), 0, np.cos(yaw_angle), 0],
                      [0, 0, 0, 1]])
    return R_yaw

def create_roll_rotation_matrix(roll_angle):
    """
    Create a rotation matrix for roll rotation.

    Parameters:
    - roll_angle: Angle in radians for roll rotation.

    Returns:
    - R_roll: Rotation matrix for roll.
    """
    R_roll = np.array([[np.cos(roll_angle), -np.sin(roll_angle), 0, 0],
                       [np.sin(roll_angle), np.cos(roll_angle), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
    return R_roll

def affine_augment(vertices):
    
    # Add affine matrix augmentations
    # Generate random values for rotation angles with a range of plus or minus 2 degrees
    theta_x_deg = np.random.uniform(-2.0, 2.0)  # Generate random values in degrees
    theta_y_deg = np.random.uniform(-2.0, 2.0)
    theta_z_deg = np.random.uniform(-2.0, 2.0)

    # Convert degrees to radians
    theta_x = np.deg2rad(theta_x_deg)
    theta_y = np.deg2rad(theta_y_deg)
    theta_z = np.deg2rad(theta_z_deg)
    
    # Generate random values for scaling factors
    scale_x = np.random.uniform(0.9, 1.1)
    scale_y = np.random.uniform(0.9, 1.1)
    scale_z = np.random.uniform(0.9, 1.1)
    
    # Generate random values for shearing factors
    shear_xy = np.random.uniform(-0.012, 0.012)
    shear_xz = np.random.uniform(-0.012, 0.012)
    shear_yx = np.random.uniform(-0.012, 0.012)
    shear_yz = np.random.uniform(-0.012, 0.012)
    shear_zx = np.random.uniform(-0.012, 0.012)
    shear_zy = np.random.uniform(-0.012, 0.012)

    # Create rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    # Create scaling matrices
    Sx = np.array([[scale_x, 0, 0], [0, 1, 0], [0, 0, 1]])
    Sy = np.array([[1, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    Sz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, scale_z]])
    
    # Create shearing matrix
    Sh = np.array([[1, shear_xy, shear_xz], [shear_yx, 1, shear_yz], [shear_zx, shear_zy, 1]])

    # Create the combined transformation matrix T
    T = Sx @ Sy @ Sz @ Rx @ Ry @ Rz @ Sh

    affine_vertices = np.dot(vertices, T)
    return affine_vertices

def elastic_augment(vertices):
    
    box = [2, 3, 4, 5] # I dont add the tuple with 1,0 (sigma 0) because it is going to bias towards that
                                                        # since the sigma is already multiplyed bu rand(1)
    
    # Randomly pick a tuple
    box_size = random.choice(box)
    sigma = 5
    sigma *= np.random.rand(1)[0]
    
    # Eugenio's version
    fieldLR =sigma * np.random.randn(box_size,box_size,box_size,3)

    mini = np.min(vertices, axis=0)
    maxi = np.max(vertices, axis=0)

    vertices_norm = (vertices - mini) / (maxi - mini)
    
    v = np.arange(0, box_size)
    interp = rgi((v, v, v), fieldLR, method='linear', bounds_error=False, fill_value=None)
    displacements = interp(vertices_norm)
    elastic_vertices = vertices + displacements
    return elastic_vertices

    
def validate_files(input_path):
    l_paths = []
    m_paths = []

    labels_path = os.path.join(input_path, 'label')
    surf_path = os.path.join(input_path, 'surf')

    if os.path.exists(labels_path):
        for root, dirs, files in os.walk(labels_path):
            for file in files:
                l_paths.append(os.path.join(root, file))

    if os.path.exists(surf_path):
        for root, dirs, files in os.walk(surf_path):
            for file in files:
                m_paths.append(os.path.join(root, file))

    # Create dictionaries to associate names with paths
    dict_l = {path.split('/')[-1].split('.')[0]: path for path in l_paths}
    dict_m = {path.split('/')[-1].split('.')[0]: path for path in m_paths}

    # Find common names between the two dictionaries
    common_names = set(dict_l) & set(dict_m)

    # Validate paths for common names
    for name in common_names:
        l_path = dict_l.get(name)
        m_path = dict_m.get(name)

        # Check if paths are valid for corresponding names
        if not (l_path and m_path):
            print(f"Error: Mismatched paths for name '{name}'")
            # Handle the mismatch, such as raising an exception or returning an error message
            return None

        # Check if paths correspond correctly for the names
        if not (l_path.replace('label', 'surf').replace('.annot', '.white') == m_path and
                m_path.replace('surf', 'label').replace('.white', '.annot') == l_path):
            print(f"Error: Mismatched paths for name '{name}'")
            # Handle the mismatch, such as raising an exception or returning an error message
            return None

    # Sort paths based on the order of names
    l_paths = [dict_l[name] for name in sorted(common_names)]
    m_paths = [dict_m[name] for name in sorted(common_names)]

    print('Files checked!')
    
    return sorted(common_names), m_paths, l_paths



'''
def validate_files(input_path):
    # Use os.listdir() to get a list of all items (files and directories) in the specified path
    names = os.listdir(input_path)
    valid_names = []
    mesh_paths = []
    labels_paths = []

    print('Checking if files exist...')
    #log_progress('Checking if files exist...')

    for name in names:
        mesh_path = os.path.join(input_path, name, 'surf', 'lh_aligned.surf')
        labels_path = os.path.join(input_path, name, 'label', 'lh.annot')

        if os.path.exists(mesh_path) and os.path.exists(labels_path):
            mesh_paths.append(mesh_path)
            labels_paths.append(labels_path)
            valid_names.append(name)
        else:
            if not os.path.exists(mesh_path):
                print(f"Mesh file not found for subject '{name}'.")
                #log_progress(f"Mesh file not found for subject '{name}'.")

            if not os.path.exists(labels_path):
                print(f"Labels file not found for subject '{name}'.")
                #log_progress(f"Labels file not found for subject '{name}'.")

            print(f"Skipping '{name}' due to missing file(s).")
            #log_progress(f"Skipping '{name}' due to missing file(s).")

    names = valid_names
    print('Files checked!')
    #log_progress('Files checked!')
    
    # Check if mesh_paths and label_paths have the same length
    assert len(mesh_paths) == len(labels_paths), "Error: The number of mesh paths and label paths does not match."

    return names, mesh_paths, labels_paths




# Function to find an available log filename
def find_available_log_filename(base_filename):
    counter = 0
    while True:
        counter += 1
        log_filename = f"{base_filename}_{counter:02d}.txt"
        if not os.path.exists(log_filename):
            return log_filename

# Function to log progress
def log_progress(message):
    if log_file_path is not False:
        with open(log_file_path, "a") as log_file:
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
            log_file.write(timestamp + message + "\n")
'''

