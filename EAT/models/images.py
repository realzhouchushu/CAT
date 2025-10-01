# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from timm.models.layers import to_2tuple
from fairseq.tasks import FairseqTask
from enum import Enum, auto
from .vision_transformer import CBlock

from .mae import PatchEmbed,get_2d_sincos_pos_embed_flexible,PatchEmbed_new


from .base import (
    D2vModalityConfig,
    ModalitySpecificEncoder,
    get_alibi_bias,
    MaskSeed,
)
from .modules import (
    BlockEncoder,
    Decoder2d,
    FixedPositionalEncoder,
    TransformerDecoder,
    EncDecTransformerDecoder,
)


class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()

logger = logging.getLogger(__name__)


@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768
    conv_option: int = 0

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False
    target_length: int = 1024
    max_length: int = 768

class ImageEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task: Optional[FairseqTask],
        add_conv,
    ):
        
        if modality_cfg.in_chans == 1 :  
            img_size = (modality_cfg.target_length,128)
        else:
            img_size =  to_2tuple(modality_cfg.input_size)

        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # number of patch -> 512
        self.H = img_size[0] // patch_size[0]  # 64
        self.W = img_size[1] // patch_size[1]  # 8
        self.hw = (self.H,self.W)

        local_encoder = None
        patch_embed = None
        stage_output_decode = None
        conv_blocks = None

        # # (B,512,768)
        # # note: we fix the variable length sequence problem here -> not limited to fixed length data
        if not add_conv:
            local_encoder = PatchEmbed_new(
                img_size,
                modality_cfg.patch_size,
                modality_cfg.in_chans,
                modality_cfg.embed_dim,
                add_conv=add_conv,
            )

            # CNN initialize
            w = local_encoder.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            if modality_cfg.embed_dim != embed_dim:
                local_encoder = nn.Sequential(
                    local_encoder,
                    nn.Linear(modality_cfg.embed_dim, embed_dim),
                )
            conv_params = None
        else:
            # manual setting
            if modality_cfg.conv_option == 0:
                resolution = [4, 8, 16]
                in_chans = [modality_cfg.in_chans, 256, 384, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 1:
                resolution = [16]
                in_chans = [modality_cfg.in_chans, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 2:
                resolution = [4, 8]
                in_chans = [modality_cfg.in_chans, 384, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 3:
                resolution = [4, 16]
                in_chans = [modality_cfg.in_chans, 384, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 4:
                resolution = [8, 16]
                in_chans = [modality_cfg.in_chans, 384, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 5:
                resolution = [2, 4, 8, 16]
                in_chans = [modality_cfg.in_chans, 256, 384, 768, modality_cfg.embed_dim]
            elif modality_cfg.conv_option == 6:
                resolution = [4, 8, 16, 32]
                in_chans = [modality_cfg.in_chans, 256, 384, 768, modality_cfg.embed_dim]
            else:
                raise ValueError(f"Invalid conv option: {modality_cfg.conv_option}")
            
            patch_sizes = []
            downsample_rate = [1]
            stage_output_patch_sizes = []
            depth = [2 for _ in range(len(resolution) - 1)]
            mlp_ratio = [4 for _ in range(len(resolution) - 1)]
            for i, res in enumerate(resolution):
                patch_size = resolution[i] / resolution[i - 1] if i else resolution[i]
                patch_sizes.append(int(patch_size))
                
                if i:
                    downsample_rate.append(patch_sizes[i - 1] * downsample_rate[-1])
            for i in range(len(patch_sizes) - 1):
                stage_output_patch_size = (patch_sizes[len(patch_sizes) - 1 - i] * stage_output_patch_sizes[-1]) if i else patch_sizes[len(patch_sizes) - 1 - i]
                stage_output_patch_sizes.append(stage_output_patch_size)
            stage_output_patch_sizes = list(reversed(stage_output_patch_sizes))
            logger.info(f"resolution: {resolution}")
            logger.info(f"in_chans: {in_chans}")
            logger.info(f"patch_sizes: {patch_sizes}")
            logger.info(f"downsample_rate: {downsample_rate}")
            logger.info(f"depth: {depth}")
            logger.info(f"mlp_ratio: {mlp_ratio}")
            logger.info(f"stage_output_patch_sizes: {stage_output_patch_sizes}")
            conv_params = {
                "patch_sizes": patch_sizes,
                "downsample_rate": downsample_rate,
                "depth": depth,
                "mlp_ratio": mlp_ratio,
                "stage_output_patch_sizes": stage_output_patch_sizes,
            }
            # patch_sizes = [4, 2, 2]
            # downsample_rate = [1, 4, 8]
            # depth = [2, 2]
            # mlp_ratio=[4, 4]
            patch_embed = [
                PatchEmbed_new(
                img_size=(img_size[0]//downsample_rate[i], img_size[1]//downsample_rate[i]),
                patch_size=patch_sizes[i],
                in_chans=in_chans[i],
                embed_dim=in_chans[i+1],
                stride=patch_sizes[i],
                add_conv=add_conv,
            ) for i in range(len(downsample_rate))]

            patch_embed.append(nn.Linear(modality_cfg.embed_dim, modality_cfg.embed_dim))
            patch_embed = nn.ModuleList(patch_embed)

            stage_output_decode = nn.ModuleList([nn.Conv2d(in_chans[i+1], in_chans[-1], stage_output_patch_sizes[i], stride=stage_output_patch_sizes[i]) for i in range(len(stage_output_patch_sizes))])

            conv_blocks = []
            for i in range(len(mlp_ratio)):
                logger.info(f"i: {i}")
                conv_blocks.append(
                    nn.ModuleList([CBlock(dim=in_chans[i+1],  mlp_ratio=mlp_ratio[i], norm_layer=partial(nn.LayerNorm, eps=1e-6))
                    for j in range(depth[i]) ])
                )
            conv_blocks = nn.ModuleList(conv_blocks)

            # CNN initialize
            for m in patch_embed:
                if hasattr(m, 'proj'):
                    w = m.proj.weight.data
                else:
                    w = m.weight.data
                torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        project_features = nn.Identity()

        # note: max_length control the maximum time length of audio -> "64" for 10s, here we define it as 2min, you can change it yourself
        max_length = modality_cfg.max_length
        pos_embed = nn.Parameter(
            torch.zeros(1, max_length*self.W, embed_dim), requires_grad=False
        )

        # side_n = int(num_patches ** 0.5)
        # note: we fix the variable length sequence problem here -> support up to 2min audio 
        emb = get_2d_sincos_pos_embed_flexible(
            pos_embed.shape[-1],
            (max_length,self.W),  
            cls_token=False,
        )
        
        pos_embed.data.copy_(torch.from_numpy(emb[:max_length*self.W,:]).float().unsqueeze(0)) 
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        # EAT utilize the CNN decoder
        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_enc = BlockEncoder(
                    nn.ModuleList(
                        make_block(0, modality_cfg.decoder.decoder_dim, 8)
                        for _ in range(modality_cfg.decoder.decoder_layers)
                    ),
                    None,
                    layer_norm_first,
                    0,
                    0,
                )
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder2d(modality_cfg.decoder, embed_dim, self.H, self.W)
                if modality_cfg.decoder is not None
                else None
            )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            patch_embed=patch_embed,
            stage_output_decode=stage_output_decode,
            conv_blocks=conv_blocks,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
            add_conv=add_conv,
            conv_params=conv_params,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)   audio: (N,1,H,W)   1024/16 = 64   128/16 = 8
        x: (N, L, patch_size**2 *3)
        """
        if self.modality_cfg.in_chans == 1:
            p = self.modality_cfg.patch_size
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            #h,w = self.patch_embed.patch_hw
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            
        else:
            p = self.modality_cfg.patch_size
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            from ..utils.data_utils import compute_block_mask_2d

            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape

            mask = compute_block_mask_2d(
                shape=(B, L),
                mask_prob=self.modality_cfg.mask_prob,
                mask_length=self.modality_cfg.mask_length,
                mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
                inverse_mask=self.modality_cfg.inverse_mask,
                require_same_masks=True,
                mask_dropout=self.modality_cfg.mask_dropout,
                img_shape=self.hw
            )
            

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        if (
            not self.modality_cfg.transformer_decoder
            or not self.modality_cfg.enc_dec_transformer
        ):
            return super().decoder_input(x, mask_info)

        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        kv = x[:, self.modality_cfg.num_extra_tokens :]

        assert self.fixed_positional_encoder is not None
        pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)

        mask = mask_info.mask.bool()
        if self.modality_cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)

        q = pos[mask].view(x.size(0), -1, x.size(-1))

        return q, kv
