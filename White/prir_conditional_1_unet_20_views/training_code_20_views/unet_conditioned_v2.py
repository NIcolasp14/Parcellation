"""
Improved U-Net with ONE-HOT VIEW CONDITIONING
==============================================
Instead of continuous camera parameters, uses discrete 20-way one-hot encoding:
- Views 0-5: Cardinal views (Front, Back, Left, Right, Top, Bottom)
- Views 6-9: Corner views
- Views 10-13: Top diagonal views
- Views 14-17: Bottom diagonal views
- Views 18-19: Side-top views

This is more robust and interpretable than camera parameters.
"""

import torch
import torch.nn as nn

class ViewConditionedUNet(nn.Module):
    """U-Net conditioned on view type via one-hot encoding"""
    
    def __init__(self, in_channels=3, num_classes=37, num_views=20):
        super(ViewConditionedUNet, self).__init__()
        
        self.num_views = num_views
        
        # View embedding: one-hot → learned embedding
        self.view_embedding = nn.Sequential(
            nn.Linear(num_views, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
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
    
    def forward(self, x, view_onehot):
        """
        Args:
            x: (B, 3, H, W) - input images
            view_onehot: (B, 20) - one-hot encoded view labels
        """
        # Encode view
        view_emb = self.view_embedding(view_onehot)  # (B, 128)
        
        # Encoder
        e1 = self.enc1(x)       # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 512, H/8, W/8)
        
        # Bottleneck: inject view embedding
        # Pool e4 first
        e4_pooled = self.pool(e4)  # (B, 512, H/16, W/16)
        
        # Expand view embedding to match pooled spatial dimensions
        B, C, H, W = e4_pooled.shape
        view_spatial = view_emb.view(B, 128, 1, 1).expand(B, 128, H, W)
        
        # Concatenate encoder output with view embedding
        bottleneck_input = torch.cat([e4_pooled, view_spatial], dim=1)  # (B, 512+128, H/16, W/16)
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
    model = ViewConditionedUNet(in_channels=3, num_classes=37, num_views=20)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 800, 800)
    
    # Test one-hot view labels
    view_labels = torch.tensor([0, 1, 2, 3])  # Front, Back, Left, Right
    view_onehot = torch.zeros(batch_size, 20)
    view_onehot[torch.arange(batch_size), view_labels] = 1
    
    # Forward pass
    output = model(x, view_onehot)
    
    print(f"Input shape: {x.shape}")
    print(f"View one-hot shape: {view_onehot.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == '__main__':
    test_model()

