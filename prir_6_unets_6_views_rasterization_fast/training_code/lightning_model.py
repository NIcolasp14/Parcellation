#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Lightning model


# In[3]:


from unetxd import UNet2d
import torch
from torch import nn
import pytorch_lightning as pl

# Import the UNet model and other necessary modules

# Define a PyTorch Lightning module
class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(UNetLightning, self).__init__()
        self.unet_model = UNet2d(in_channels=in_channels, out_channels=out_channels, skip=True, convs_per_block=2)  # Use the UNet model
        self.save_hyperparameters()

    def forward(self, x):
        # Forward pass through the UNet model
        return self.unet_model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)#, verbose = True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        train_loss = nn.CrossEntropyLoss()(outputs, labels)
        
        self.log('training_loss_epoch', train_loss, on_step=False, on_epoch=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)

        self.log("validation_loss", val_loss, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)


        self.log("test_loss", test_loss)

        return test_loss

'''
Example usage
model = UNetLightning(in_channels = 3, out_channels = 3)
'''

