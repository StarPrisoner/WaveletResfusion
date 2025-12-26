import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet import Unet
from .UNet_parts import *  # Assume standard UNet components are here
from utils.dwt import DWTForward


# =============================================================================
# 1. Core Block: Hierarchical Wavelet Block (HWB)
# =============================================================================

class HierarchicalWaveBlock(nn.Module):
    """
    HWB: Decomposes features into wavelet subbands and processes
    high-frequency components through directional branches.
    """

    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.dwt = DWTForward()

        # Subband branches: Low frequency vs. Directional High frequency
        self.low_freq_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.high_freq_branches = nn.ModuleList([
            DirectionalConvBranch(dim, d) for d in ["horizontal", "vertical", "diagonal"]
        ])

        # Feature fusion and dynamic gating
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(4 * dim, 4 * dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(4 * dim, 4 * dim, 1),
            ChannelSplitActivation(dim)
        )
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape
        ll, hf = self.dwt(x)
        lh, hl, hh = [h.squeeze(2) for h in hf[0].chunk(3, dim=2)]

        # Frequency-aware feature extraction
        feats = [self.low_freq_conv(ll)] + [branch(h) for branch, h in zip(self.high_freq_branches, [lh, hl, hh])]
        feats = [F.interpolate(f, size=(H, W), mode='bilinear') for f in feats]

        concated = torch.cat(feats, dim=1)
        gates = self.fusion_gate(concated)

        fused = 0
        for i in range(4):
            fused += feats[i] * gates[:, i * C:(i + 1) * C]

        return x + self.res_scale * fused


# =============================================================================
# 2. Core Block: Cross-Scale Fusion (CSF)
# =============================================================================

class CrossScaleFusion(nn.Module):
    """CSF: Aligns and fuses features across different scales of the WaveUnet"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attn = HybridAttention(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )
        self.fusion_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, prev_feat, curr_feat):
        if prev_feat.shape[2:] != curr_feat.shape[2:]:
            prev_feat = F.interpolate(prev_feat, size=curr_feat.shape[2:], mode='bilinear')
        fused = self.attn(self.conv(torch.cat([prev_feat, curr_feat], dim=1)))
        return curr_feat + self.fusion_scale * fused


# =============================================================================
# 3. Main Architecture: WaveUnet (Full Implementation)
# =============================================================================

class WaveUnet(Unet):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), **kwargs):
        super().__init__(dim=dim, dim_mults=dim_mults, **kwargs)
        self.time_emb_dim = dim * 4

        # Initialize custom HWB modules for each encoder stage
        self.wave_blocks = nn.ModuleList([
            HierarchicalWaveBlock(dim * m, self.time_emb_dim) for m in dim_mults
        ])

        # Initialize CSF modules for the decoder pathway
        down_dims = [dim * m for m in dim_mults]
        up_dims = list(reversed(down_dims))[1:]
        self.cross_fusions = nn.ModuleList([
            CrossScaleFusion(in_channels=down_dims[-1 - i] + up_dims[i], out_channels=up_dims[i])
            for i in range(len(up_dims))
        ])

    def forward(self, x, time, input_cond=None):
        # 1. Initial Processing and Condition Injection
        if input_cond is not None:
            x = torch.cat((x, input_cond), dim=1)
        x = self.init_conv(x)
        residual, t = x.clone(), self.time_mlp(time)

        # 2. Encoder: Standard blocks + HWB
        down_features, pyramid_features = [], []
        for wave_block, down_stage in zip(self.wave_blocks, self.downs):
            x = down_stage[0](x, t)  # Resnet block
            pyramid_features.append(x)
            x = down_stage[1](x, t)  # Resnet block
            pyramid_features.append(x)

            x = down_stage[-1](x)  # Downsampling
            x = wave_block(x, t)  # Hierarchical Wavelet Processing
            down_features.append(x)

        # 3. Bottleneck
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t)), t)

        # 4. Decoder: CSF integration during upsampling
        for idx, (block1, block2, attn, upsample) in enumerate(self.ups):
            for block in [block1, block2]:
                skip = pyramid_features.pop()
                x = torch.cat((x, skip), dim=1)
                x = block(x, t)

            x = upsample(attn(x))
            # Apply Cross-Scale Fusion between encoder and decoder features
            if idx < len(self.cross_fusions):
                x = self.cross_fusions[idx](down_features[len(self.wave_blocks) - 1 - idx], x)

        # 5. Output Head
        x = self.final_res_block(torch.cat([x, residual], dim=1), t)
        return self.final_conv(x)