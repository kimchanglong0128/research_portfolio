# TimeEmbedding + Residual Block + Attention Block + Upsample + Downsample implementations for diffusion models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Time Embedding Module
class TimeEmbedding(nn.Module):
    """
    t -> time embedding
    sin / cose + MLP
    """
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # [half_dim]
        half_dim = self.dim // 2        # sin/cos embedding dimension of B
        emb = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        # [B, 1] * [half_dim] -> [B, half_dim]
        args = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        # [B, dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # MLP
        emb = self.mlp(emb)
        return emb

# Residual Block
class ResBlock(nn.Module):
    """
    - conv -> norm -> SiLU -> conv -> norm
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 32):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        t: [B, time_dim]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t_add = self.time_mlp(t_emb) # [B, out_ch]
        t_add = t_add[:, :, None, None]  # [B, out_ch, 1, 1]
        h = h + t_add 

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.skip(x)
    

# Multi-CrossAttention Block
class AttentionBlock(nn.Module):
    """
    - x: [B, C, H, W] <- Unet feature map (H*W flattened as sequence)
    - cond: [B, L, cond_dim] <- Clip embedding 
    """
    def __init__(self, dim: int, cond_dim: int, num_heads: int = 4):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_heads = num_heads

        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(cond_dim, dim)
        self.to_v = nn.Linear(cond_dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        cond: [B, L, cond_dim]
        """
        B, N, C = x.shape
        _, L, _ = cond.shape

        # linear projections
        q = self.to_q(x)  # [B, N, C]
        k = self.to_k(cond)  # [B, L, C]
        v = self.to_v(cond)  # [B, L, C]

        # Multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, L]
        attn = torch.softmax(attn, dim=-1)  # [B, num
        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        out = self.proj(out)  # [B, N, C]
        return out
    

# Upsample Block
class Downsample(nn.Module):
    """
    - 2x downsample : stride=2 conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """
    - 2x upsample : stride=2 conv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)