"""
Improved U-Net with CONTINUOUS CAMERA POSE CONDITIONING
========================================================
Uses continuous camera parameters (3 rotation angles) for arbitrary view support:
- rx, ry, rz: Rotation angles in normalized [-1, 1] range
This allows the network to generalize to any camera viewpoint.
"""

import torch
import torch.nn as nn

class ViewConditionedUNet(nn.Module):
    """U-Net conditioned on continuous camera pose (rotation angles)"""
    
    def __init__(self, in_channels=3, num_classes=37, pose_dim=3):
        super(ViewConditionedUNet, self).__init__()
        
        self.pose_dim = pose_dim
        
        # Camera pose embedding: continuous parameters → learned embedding
        self.pose_embedding = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Encoder with view conditioning
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck with view conditioning
        # Add 128 channels for view embedding
        self.bottleneck = self.conv_block(512 + 128, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final classifier
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, camera_pose):
        """
        Args:
            x: (B, 3, H, W) - input images
            camera_pose: (B, 3) - normalized rotation angles [rx, ry, rz] in [-1, 1]
        """
        # Encode camera pose
        pose_emb = self.pose_embedding(camera_pose)  # (B, 128)
        
        # Encoder
        e1 = self.enc1(x)       # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 512, H/8, W/8)
        
        # Bottleneck: inject camera pose embedding
        # Pool e4 first
        e4_pooled = self.pool(e4)  # (B, 512, H/16, W/16)
        
        # Expand pose embedding to match pooled spatial dimensions
        B, C, H, W = e4_pooled.shape
        pose_spatial = pose_emb.view(B, 128, 1, 1).expand(B, 128, H, W)
        
        # Concatenate encoder output with pose embedding
        bottleneck_input = torch.cat([e4_pooled, pose_spatial], dim=1)  # (B, 512+128, H/16, W/16)
        b = self.bottleneck(bottleneck_input)  # (B, 1024, H/16, W/16)
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        
        return out


def test_model():
    """Test the model architecture"""
    model = ViewConditionedUNet(in_channels=3, num_classes=37, pose_dim=3)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 800, 800)
    
    # Test camera pose (normalized rotation angles)
    camera_pose = torch.randn(batch_size, 3) * 0.5  # Random poses in [-0.5, 0.5]
    
    # Forward pass
    output = model(x, camera_pose)
    
    print(f"Input shape: {x.shape}")
    print(f"Camera pose shape: {camera_pose.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    test_model()

