import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from einops import rearrange  

class TransformerCnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, drop_prob=0.1):
        super(TransformerCnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=drop_prob)

    def forward(self, x):
        out = self.conv1(x)
        attn = self.attention(out)
        out = out * attn
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.drop(out)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=16, dropout_rate=0.1):
        super(SpatialAttentionBlock, self).__init__()
        self.multihead_attn_spatial = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm1_spatial = nn.LayerNorm(dim)
        self.norm2_spatial = nn.LayerNorm(dim)
        self.fc_spatial = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        N, C, H, W = x.size()
        x_flat = rearrange(x, 'n c h w -> (h w) n c')

        attn_output, _ = self.multihead_attn_spatial(x_flat, x_flat, x_flat)
        attn_output = self.dropout(attn_output)

        x = self.norm1_spatial(x_flat + attn_output)

        ffn_output = self.fc_spatial(x)
        ffn_output = self.dropout(ffn_output)

        x = self.norm2_spatial(x + ffn_output)

        x = rearrange(x, '(h w) n c -> n c h w', h=H, w=W)
        return x


class CrossSpatialAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=16, dropout_rate=0.1):
        super(CrossSpatialAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_query, x_key_value):
        N, C, H, W = x_query.size()

        xq = rearrange(x_query, 'n c h w -> (h w) n c')  # [H*W, N, C]
        xkv = rearrange(x_key_value, 'n c h w -> (h w) n c')  # [H*W, N, C]

        attn_output, _ = self.multihead_attn(xq, xkv, xkv)
        attn_output = self.dropout(attn_output)

        x = self.norm1(xq + attn_output)

        ffn_output = self.fc(x)
        ffn_output = self.dropout(ffn_output)

        x = self.norm2(x + ffn_output)

        x = rearrange(x, '(h w) n c -> n c h w', h=H, w=W)
        return x
    
class DGIQA(nn.Module):
    def __init__(self, img_size=224):
        super(DGIQA, self).__init__()

        # Swin Transformer based RGB Backbone
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            features_only=True,
            img_size=img_size
        )

        # Swin Transformer based Depth Backbone
        self.depth_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            features_only=True,
            img_size=img_size
        )

        # Transformer-CNN Bridges for RGB
        self.tcb1 = TransformerCnnBlock(in_channels=96, out_channels=64)
        self.tcb2 = TransformerCnnBlock(in_channels=192, out_channels=128)
        self.tcb3 = TransformerCnnBlock(in_channels=384, out_channels=256)
        self.tcb4 = TransformerCnnBlock(in_channels=768, out_channels=512)

        # Transformer-CNN Bridges for Depth
        self.tcb1_depth = TransformerCnnBlock(in_channels=96, out_channels=64)
        self.tcb2_depth = TransformerCnnBlock(in_channels=192, out_channels=128)
        self.tcb3_depth = TransformerCnnBlock(in_channels=384, out_channels=256)
        self.tcb4_depth = TransformerCnnBlock(in_channels=768, out_channels=512)


        # Depth-CAR Blocks
        self.depth_guided_cross_attention = CrossSpatialAttentionBlock(dim=960, num_heads=16)
        self.refinement_attention = SpatialAttentionBlock(dim=960, num_heads=16)

        # Dilated Convolutional Stack
        self.dilated_stack = nn.Sequential(
            nn.Conv2d(960, 1024, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Global Pooling and Fully Connected Layer
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, 1)

    def forward(self, x_rgb, x_depth):
        N, C, H, W = x_rgb.size()

        features_rgb = self.backbone(x_rgb)
        features_depth = self.depth_backbone(x_depth)

        if len(features_rgb) >= 4:
            x1 = features_rgb[0]
            x2 = features_rgb[1]
            x3 = features_rgb[2]
            x4 = features_rgb[3]

            x1_depth = features_depth[0]
            x2_depth = features_depth[1]
            x3_depth = features_depth[2]
            x4_depth = features_depth[3]
        else:
            raise ValueError(f"Expected at least 4 feature maps from backbone, got {len(features_rgb)}")

        # Rearrange feature maps from [N, H, W, C] to [N, C, H, W]
        x1 = rearrange(x1, 'n h w c -> n c h w')
        x2 = rearrange(x2, 'n h w c -> n c h w')
        x3 = rearrange(x3, 'n h w c -> n c h w')
        x4 = rearrange(x4, 'n h w c -> n c h w')

        x1_depth = rearrange(x1_depth, 'n h w c -> n c h w')
        x2_depth = rearrange(x2_depth, 'n h w c -> n c h w')
        x3_depth = rearrange(x3_depth, 'n h w c -> n c h w')
        x4_depth = rearrange(x4_depth, 'n h w c -> n c h w')

        # Process each scale for RGB
        s1 = self.tcb1(x1)
        s2 = self.tcb2(x2)
        s3 = self.tcb3(x3)
        s4 = self.tcb4(x4)

        # Process each scale for Depth
        s1_depth = self.tcb1_depth(x1_depth)
        s2_depth = self.tcb2_depth(x2_depth)
        s3_depth = self.tcb3_depth(x3_depth)
        s4_depth = self.tcb4_depth(x4_depth)

        # Resize all to the smallest spatial dimensions
        min_H = min(s1.size(2), s2.size(2), s3.size(2), s4.size(2))
        min_W = min(s1.size(3), s2.size(3), s3.size(3), s4.size(3))

        s1 = F.interpolate(s1, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s4 = F.interpolate(s4, size=(min_H, min_W), mode='bilinear', align_corners=False)

        s1_depth = F.interpolate(s1_depth, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s2_depth = F.interpolate(s2_depth, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s3_depth = F.interpolate(s3_depth, size=(min_H, min_W), mode='bilinear', align_corners=False)
        s4_depth = F.interpolate(s4_depth, size=(min_H, min_W), mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        fused_rgb = torch.cat([s1, s2, s3, s4], dim=1)       # [N, 960, min_H, min_W]
        fused_depth = torch.cat([s1_depth, s2_depth, s3_depth, s4_depth], dim=1)  # [N, 960, min_H, min_W]
        
        # Depth-Guided Cross Attention
        fused = self.depth_guided_cross_attention(fused_depth, fused_rgb) 

        # Refinement: Self-Attention Blocks
        fused = self.refinement_attention(fused)

        # Dilated Convolutional Stack
        fused = self.dilated_stack(fused)

        # Global Pooling and Fully Connected Layer
        pooled = self.global_pool(fused)
        flattened = torch.flatten(pooled, 1)
        out = self.fc(flattened)
        out = torch.sigmoid(out)
        out = out.view(-1)
        return out