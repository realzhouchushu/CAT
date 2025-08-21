import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Optional

# 模拟必要的类和函数
class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()

@dataclass
class D2vModalityConfig:
    type: Modality = Modality.IMAGE
    prenet_depth: int = 4
    prenet_layerdrop: float = 0
    prenet_dropout: float = 0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_extra_tokens: int = 0
    init_extra_token_zero: bool = True
    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    inverse_mask: bool = False
    mask_prob_adjust: float = 0
    keep_masked_pct: float = 0
    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True
    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64
    ema_local_encoder: bool = False
    local_grad_mult: float = 1.0
    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False
    num_alibi_heads: int = 12
    model_depth: int = 12

# 模拟DropPath
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# 模拟CMlp
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 模拟CBlock
class CBlock(nn.Module):
    def __init__(self, dim, num_heads=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x

# 模拟PatchEmbed_new
class PatchEmbed_new(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x = x.flatten(2).transpose(1, 2)
        return x

# 模拟StageOutputDecode
class StageOutputDecode(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride)
    
    def forward(self, x):
        return self.conv(x)

# 模拟ModalitySpecificEncoder
class ModalitySpecificEncoder(nn.Module):
    def __init__(self, modality_cfg, embed_dim=768):
        super().__init__()
        self.modality_cfg = modality_cfg
        self.local_grad_mult = modality_cfg.local_grad_mult
        
        # 参考 images.py 的配置
        downsample_rate = [1, 4, 8]
        patch_sizes = [4, 2, 2]
        in_chans = [1, 256, 384, embed_dim]  # 音频频谱图：1通道
        depth = [2, 2]
        mlp_ratio = [4, 4]
        
        # 模拟patch_embed (4个阶段)
        self.patch_embed = nn.ModuleList([
            PatchEmbed_new(
                img_size=(1024//downsample_rate[0], 128//downsample_rate[0]),  # (1024, 128)
                patch_size=patch_sizes[0],  # 4
                in_chans=in_chans[0],       # 1
                embed_dim=in_chans[1],      # 256
                stride=patch_sizes[0]       # 4
            ),
            PatchEmbed_new(
                img_size=(1024//downsample_rate[1], 128//downsample_rate[1]),  # (256, 32)
                patch_size=patch_sizes[1],  # 2
                in_chans=in_chans[1],       # 256
                embed_dim=in_chans[2],      # 384
                stride=patch_sizes[1]       # 2
            ),
            PatchEmbed_new(
                img_size=(1024//downsample_rate[2], 128//downsample_rate[2]),  # (128, 16)
                patch_size=patch_sizes[2],  # 2
                in_chans=in_chans[2],       # 384
                embed_dim=in_chans[3],      # 768
                stride=patch_sizes[2]       # 2
            ),
            nn.Linear(embed_dim, embed_dim)  # 最后的线性层
        ])
        
        # 模拟conv_blocks (2个阶段)
        self.conv_blocks = nn.ModuleList([
            nn.ModuleList([CBlock(dim=in_chans[1], mlp_ratio=mlp_ratio[0], norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(depth[0])]),  # 阶段0: 2层，256通道
            nn.ModuleList([CBlock(dim=in_chans[2], mlp_ratio=mlp_ratio[1], norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(depth[1])]),  # 阶段1: 2层，384通道
        ])
        
        # 模拟stage_output_decode (2个阶段)
        self.stage_output_decode = nn.ModuleList([
            StageOutputDecode(in_channels=in_chans[1], out_channels=embed_dim, patch_size=patch_sizes[0], stride=patch_sizes[0]),  # 256->768, 4x4
            StageOutputDecode(in_channels=in_chans[2], out_channels=embed_dim, patch_size=patch_sizes[1], stride=patch_sizes[1]),  # 384->768, 2x2
        ])

    def local_encoder(self, features, precomputed_mask):
        print(f"\n=== 开始 local_encoder 测试 ===")
        print(f"输入 features 形状: {features.shape}")
        print(f"输入 precomputed_mask 形状: {precomputed_mask.shape if precomputed_mask is not None else 'None'}")
        
        # Stage 0
        print(f"\n--- Stage 0 ---")
        x = self.patch_embed[0](features)
        print(f"patch_embed[0] 输出形状: {x.shape}")
        
        for i, blk in enumerate(self.conv_blocks[0]):
            x = blk(x)
            print(f"conv_blocks[0][{i}] 输出形状: {x.shape}")
        
        stage1_embed = self.stage_output_decode[0](x).flatten(2).permute(0, 2, 1)
        print(f"stage_output_decode[0] 输出形状: {stage1_embed.shape}")
        
        # Stage 1
        print(f"\n--- Stage 1 ---")
        x = self.patch_embed[1](x)
        print(f"patch_embed[1] 输出形状: {x.shape}")

        for i, blk in enumerate(self.conv_blocks[1]):
            x = blk(x)
            print(f"conv_blocks[1][{i}] 输出形状: {x.shape}")
        
        stage2_embed = self.stage_output_decode[1](x).flatten(2).permute(0, 2, 1)
        print(f"stage_output_decode[1] 输出形状: {stage2_embed.shape}")
        
        # Stage 2 & 3
        print(f"\n--- Stage 2 & 3 ---")
        x = self.patch_embed[2](x)
        print(f"patch_embed[2] 输出形状: {x.shape}")
        x = x.flatten(2).permute(0, 2, 1)
        
        x = self.patch_embed[3](x)  # 线性层
        print(f"patch_embed[3] (Linear) 输出形状: {x.shape}")
        
        # 最终输出
        unmasked_feature = x + stage1_embed + stage2_embed
        print(f"最终 unmasked_feature 形状: {unmasked_feature.shape}")
        
        # 处理掩码特征
        if precomputed_mask is None:
            mask_feature = None
            print(f"mask_feature: None")
        else:
            print(f"\n--- 处理掩码特征 ---")
            # 模拟掩码处理
            mask_for_patch1 = precomputed_mask.reshape(-1, 64, 8).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(-1, 64, 8, 4, 4).permute(0, 1, 3, 2, 4).reshape(precomputed_mask.shape[0], 256, 32).unsqueeze(1)
            print(f"mask_for_patch1 形状: {mask_for_patch1.shape}")
            
            mask_for_patch2 = precomputed_mask.reshape(-1, 64, 8).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(-1, 64, 8, 2, 2).permute(0, 1, 3, 2, 4).reshape(precomputed_mask.shape[0], 128, 16).unsqueeze(1)
            print(f"mask_for_patch2 形状: {mask_for_patch2.shape}")
            
            features = features.repeat_interleave(precomputed_mask.shape[0]//features.shape[0], 0)
            print(f"扩展后 features 形状: {features.shape}")
            
            # 重新计算掩码特征
            x = self.patch_embed[0](features)
            
            for i, blk in enumerate(self.conv_blocks[0]):
                x = blk(x, 1 - mask_for_patch1)
                print(f"掩码 conv_blocks[0][{i}] 输出形状: {x.shape}")
            
            stage1_embed = self.stage_output_decode[0](x).flatten(2).permute(0, 2, 1)
            print(f"掩码 stage_output_decode[0] 输出形状: {stage1_embed.shape}")
            
            x = self.patch_embed[1](x)
            
            for i, blk in enumerate(self.conv_blocks[1]):
                x = blk(x, 1 - mask_for_patch2)
                print(f"掩码 conv_blocks[1][{i}] 输出形状: {x.shape}")
            
            stage2_embed = self.stage_output_decode[1](x).flatten(2).permute(0, 2, 1)
            print(f"掩码 stage_output_decode[1] 输出形状: {stage2_embed.shape}")
            
            x = self.patch_embed[2](x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.patch_embed[3](x)
            mask_feature = x + stage1_embed + stage2_embed
            print(f"最终 mask_feature 形状: {mask_feature.shape}")
        
        return unmasked_feature, mask_feature

def main():
    print("=== ModalitySpecificEncoder local_encoder 测试脚本 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建配置
    modality_cfg = D2vModalityConfig()
    
    # 创建模型
    model = ModalitySpecificEncoder(modality_cfg, embed_dim=768)
    
    # 生成随机输入数据
    batch_size = 2
    features = torch.randn(batch_size, 1, 1024, 128)  # [B, 1, T(1024), F(128)] # dataset会padding到1024，所有音频长度一定是张这样
    print(f"输入 features 形状: {features.shape}")
    
    # 测试无掩码情况
    print("\n" + "="*50)
    print("测试无掩码情况")
    print("="*50)
    unmasked_feature, mask_feature = model.local_encoder(features, precomputed_mask=None)
    
    # 测试有掩码情况
    print("\n" + "="*50)
    print("测试有掩码情况")
    print("="*50)
    clone_batch = 3
    precomputed_mask = torch.randint(0, 2, (batch_size * clone_batch, 512), dtype=torch.float32)  # [B*cloneB, T] #patch数一定是512，因为前面音频一定是1024*128 patch大小取16的话，打出来正好是512个patch
    print(f"输入 precomputed_mask 形状: {precomputed_mask.shape}")
    
    unmasked_feature, mask_feature = model.local_encoder(features, precomputed_mask=precomputed_mask)
    
    print("\n" + "="*50)
    print("测试完成")
    print("="*50)

if __name__ == "__main__":
    main()