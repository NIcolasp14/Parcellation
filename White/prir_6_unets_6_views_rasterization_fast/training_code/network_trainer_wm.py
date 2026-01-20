#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Network trainer


# In[19]:


from utils_wm import *
from dataset_generator import *
from lightning_model import *
from lightning_datamodule import *
from monitoring_callback_6view import TrainingMonitorSingleView

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse

# Constants, change it to the folders in which the samples are
# NOTE: Using original calico storage (128x faster than lemon!)
DEFAULT_INPUT_PATH = '/autofs/vast/lemon/temp_stuff/nicolas/train_data/wm'
DEFAULT_CHECKPOINT_DIR = '/autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/Models_WM'  # Save to np341's directory


# These are the codes that link the wandb project to the training, so the trainig of each view is assocaited with a wanb project online that can rtack the metrics

rec_true_dict_wm = {
    'Top': 'eext1mwg',
    'Bottom': '3btxfs1z',
    'Right': 'p831qmrp',
    'Left': 'rt8vk2ku',
    'Back': '212ja4ip',
    'Front': '65xndsa7',
    'Random_6': 'n8d3zm6t',
    'Random': '7raxgcyo'
}

rec_false_dict_wm = {
    'Top': 'upzqqqbh',
    'Bottom': '2rkcdeiv',
    'Right': '4rjhycxv',
    'Left': 'c8qmx4tk',
    'Back': '0u6cv82w',
    'Front': 'qdyi5rs7',
    'Random_6': 'sjt3dqyw',
    'Random': '98gsca7x'
}

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('-i','--input_path', type=str, 
                    default= DEFAULT_INPUT_PATH, 
                    help="Path to the input data")
parser.add_argument('-v', '--view', type=str, required=True, help='Specify the view') # We execute one trainer code per view, however
                                                                                      # if the option 'All' is give it will create 
                                                                                      # 6 views together in the function 'generate_maps' (usually use 6 for visualization rather tahn training)
                                                                                
parser.add_argument('--recompute_normals', action='store_true', help='Flag to enable recomputing normals')
parser.add_argument('--img_width', type=int, default=800, help="Image width")
parser.add_argument('--img_height', type=int, default=800, help="Image height")
parser.add_argument('--in_channels', type=int, default=3, help="Input channels")
parser.add_argument('--out_channels', type=int, default=37, help="Output channels")
parser.add_argument('-b','--batch_size', type=int, default=8, help="Batch size")
parser.add_argument('--epochs', type=int, default=50000,
                    help="Maximum number of epochs")
parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                    help="Directory for saving model checkpoints")
parser.add_argument('--raycasting', type=str, default='pytorch3d', 
                    choices=['pytorch3d', 'open3d', 'cpu'],
                    help="Raycasting backend: 'pytorch3d' (GPU, default), 'open3d'/'cpu' (CPU fallback)")

parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help="Path to checkpoint file to resume training from")

args = parser.parse_args()

# Configure raycasting backend based on argument
import utils_wm
if args.raycasting in ['pytorch3d']:
    utils_wm.USE_GPU_RAYCASTING = True
elif args.raycasting in ['open3d', 'cpu']:
    utils_wm.USE_GPU_RAYCASTING = False
else:
    print(f"[RAYCASTING] Unknown option '{args.raycasting}', using default (PyTorch3D GPU)")
    utils_wm.USE_GPU_RAYCASTING = True

# Print GPU status AFTER configuring raycasting backend
utils_wm.print_gpu_status()

if args.recompute_normals == True:
    args.checkpoint_dir = args.checkpoint_dir.rstrip('/') + "_rec/"
    print()

print('args.checkpoint_dir: ',args.checkpoint_dir)



dirpath = os.path.join(args.checkpoint_dir, args.view)
print()
print('Dataset path: ', args.input_path)
print('Checkpoints saved in: ', dirpath)
print()





# Function for Component Setup
def setup_components(args):
    # Data Module setup
    # NOTE: num_workers=0 for GPU raycasting (PyTorch3D cannot be parallelized across processes)
    # GPU raycasting is already fast (~2s/batch), no need for multiprocessing overhead
    data_module = CustomDataModule(data_dir=args.input_path, batch_size=args.batch_size, 
                                   view=args.view, recompute_normals = args.recompute_normals, img_width=args.img_width, img_height=args.img_height,
                                   train_val_split=0.8, num_workers=0)  # 0 workers: GPU raycasting in main process
    
    # Checkpoint setup
    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        monitor="validation_loss",
        mode="min",
        dirpath=f"{args.checkpoint_dir}/{args.view}",
        filename=args.view + "-{epoch:02d}-{validation_loss:.2f}",
    )
    
    # Monitoring setup
    # Create monitoring directory based on checkpoint directory
    checkpoint_dir_base = os.path.basename(args.checkpoint_dir)
    monitoring_dir_base = f"monitoring_{checkpoint_dir_base}"
    monitoring_dir = os.path.join(os.path.dirname(args.checkpoint_dir), monitoring_dir_base, args.view)
    
    monitoring_callback = TrainingMonitorSingleView(
        monitoring_dir=monitoring_dir,
        view_name=args.view,
        log_every_n_epochs=50,  # Visualize every 50 epochs
        save_samples=True
    )
    
    print(f"Monitoring outputs for {args.view} will be saved to: {monitoring_dir}")
    
    # Wandb setup
    wandb.login(key='5754499e13af98483e7d4685f565614393d7e0af')
    if args.recompute_normals:
        # Generate the project name based on args.view
        project_name = f"{args.view}_WM_Normals_True"

        # Initialize WandB with the generated project name and ID
        print(f"Recompute normals set to {args.recompute_normals}")
        id_value = rec_true_dict_wm.get(args.view, None)
    else:
        project_name = f"{args.view}_WM_Normals_False"
        print(f"Recompute normals set to {args.recompute_normals}")
        id_value = rec_false_dict_wm.get(args.view, None)

    print(f"WandB project: {project_name}")
    print(f"View: {args.view}")
    print(f"Run ID: {id_value}")
    
    # Create WandbLogger with project name and ID - it will handle wandb.init internally
    # DISABLED: mode="disabled" to prevent disk quota errors from WandB cloud uploads
    # log_model=False to prevent checkpoint artifact logging (which writes to home dir even in disabled mode)
    # Local checkpoints are still saved by ModelCheckpoint callback!
    wandb_logger = WandbLogger(
        project=project_name,
        id=id_value,
        log_model=False,  # CRITICAL: False to avoid home dir writes
        resume="allow",
        mode="disabled"  # Disable cloud logging to avoid disk quota errors
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(dirpath, exist_ok=True)
    
    list_of_files = os.listdir(dirpath)
    checkpoint_files = [file for file in list_of_files if file.endswith(".ckpt")]
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(dirpath, f)))
        latest_checkpoint_path = os.path.join(dirpath, latest_checkpoint)
        print(f"Resuming from checkpoint: {latest_checkpoint_path}")
        # Log the latest checkpoint path and message in WandB

        model = UNetLightning.load_from_checkpoint(latest_checkpoint_path)

    else:
        model = UNetLightning(in_channels=args.in_channels, out_channels=args.out_channels)
        print('No checkpoints found...')

    trainer = pl.Trainer(
    logger=wandb_logger,
    enable_progress_bar=True,
    accelerator="auto",
    devices=[0],
    strategy="auto",
    callbacks=[checkpoint_callback, monitoring_callback],
    max_epochs=args.epochs
)


    return data_module, model, trainer

    
# Main Function

def main():
    
    data_module, model, trainer = setup_components(args)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)
    
    wandb.finish()

if __name__ == "__main__":
    main()

