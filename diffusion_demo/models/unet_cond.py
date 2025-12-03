# input -> x_t[B, 3, 32, 32] [CIFAR-10]
# t: [B]
# cond: [B, cond_dim] (CLIP embedding)

# output -> predicted noise εθ(x_t, t, cond) [B, 3, 32, 32]
# ---------------------------------------------------------------
# Down -> 2x Downsample (32 -> 16 -> 8)
# Mid with CrossAttention
# Up -> 2x Upsample (8 -> 16 -> 32)
# Skip connections between Down and Up

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Downsample, Upsample, ResBlock, AttentionBlock, TimeEmbedding

class UNetConditional(nn.Module):
    """
    Unet -> CIFAR-10 (32x32)
    - input x_t: [B, 3, 32, 32], t: [B], cond: [B, L, cond_dim]
    - output: predicted noise εθ(x_t, t, cond) [B, 3, 32, 32]
    - cond_dim: inject CLIP conditions through mid-layer cross-attention
    """
    def __init__(self, in_ch: int = 3, base_ch: int=64, cond_dim: int=512, time_dim=256, num_head: int=4):
        super(UNetConditional, self).__init__()
        self.time_mlp = TimeEmbedding(time_dim)

# -----------------------------------------------------------------------
        # Downsampling
        # 32 -> 16 -> 8
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # -> downblock ->  32 to 16
        self.down1_res1 = ResBlock(base_ch, base_ch, time_dim)
        self.down1_res2 = ResBlock(base_ch, base_ch, time_dim)
        self.down1_down = Downsample(base_ch, base_ch * 2)  # 32 -> 16

        # -> downblock -> 16 to 8
        self.down2_res1 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.down2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.down2_down = Downsample(base_ch * 2, base_ch * 4)  # 16 -> 8
# -----------------------------------------------------------------------
        # Middle
        mid_ch = base_ch * 4
        self.mid_res1 = ResBlock(mid_ch, mid_ch, time_dim)
        self.mid_attn = AttentionBlock(mid_ch, cond_dim, num_heads=num_head)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, time_dim)
# -----------------------------------------------------------------------
        # Upsampling
        # 8 -> 16 -> 32

        # -> upblock -> 8 to 16
        self.up2_up = Upsample(mid_ch, base_ch * 2)  # 8 -> 16
        self.up2_res1 = ResBlock(base_ch * 4, base_ch * 2, time_dim)  # concat skip
        self.up2_res2 = ResBlock(base_ch * 2, base_ch * 2, time_dim)

        # -> upblock -> 16 to 32
        self.up1_up = Upsample(base_ch * 2, base_ch)  # 16 -> 32
        self.up1_res1 = ResBlock(base_ch * 2, base_ch, time_dim)  # concat skip
        self.up1_res2 = ResBlock(base_ch, base_ch, time_dim)
# -----------------------------------------------------------------------
        # output
        self.out_norm = nn.GroupNorm(32, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_t:            [B, 3, 32, 32]
        t:              [B] timestep (0~T-1)
        cond_tokens:    [B, L, cond_dim]
        """
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        # Downsampling
        h = self.in_conv(x_t)  # [B, base_ch, 32, 32]

        # Down Block 1
        h = self.down1_res1(h, t_emb)
        h = self.down1_res2(h, t_emb)
        skip1 = h  # save for skip connection
        h = self.down1_down(h)  # [B, base_ch*2, 16, 16]

        # Down Block 2
        h = self.down2_res1(h, t_emb)
        h = self.down2_res2(h, t_emb)
        skip2 = h  # save for skip connection
        h = self.down2_down(h)  # [B, base_ch*4, 8, 8]

        # Middle 8 x 8 + Cross-Attention 
        h = self.mid_res1(h, t_emb)
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        attn_out = self.mid_attn(h_flat, cond_tokens)
        h_flat = attn_out  + h_flat 

        # reshape back to feature map
        h = h_flat.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        h = self.mid_res2(h, t_emb)

        # Upsampling
        # Up Block 2: 8 -> 16
        h = self.up2_up(h)  # [B, base_ch*2, 16, 16]
        # concat skip connection
        h = torch.cat([h, skip2], dim=1)  
        h = self.up2_res1(h, t_emb)
        h = self.up2_res2(h, t_emb)     # [B, base_ch*2, 16, 16]

        # Up Block 1: 16 -> 32
        h = self.up1_up(h)  # [B, base_ch, 32, 32]
        # concat skip connection
        h = torch.cat([h, skip1], dim=1)  
        h = self.up1_res1(h, t_emb)
        h = self.up1_res2(h, t_emb)     # [B, base_ch, 32, 32]

        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)  # [B, 3, 32, 32]

        return out