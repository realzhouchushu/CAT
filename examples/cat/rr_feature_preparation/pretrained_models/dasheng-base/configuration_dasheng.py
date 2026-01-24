# coding=utf-8
# Copyright 2023-2024 Xiaomi Corporation and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Dasheng model configuration"""

from transformers import PretrainedConfig

DASHENG_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mispeech/dasheng-base": "https://huggingface.co/mispeech/dasheng-base/resolve/main/config.json",
    "mispeech/dasheng-0.6B": "https://huggingface.co/mispeech/dasheng-0.6B/resolve/main/config.json",
    "mispeech/dasheng-1.2B": "https://huggingface.co/mispeech/dasheng-1.2B/resolve/main/config.json",
}


class DashengConfig(PretrainedConfig):
    model_type = "dasheng"

    def __init__(
        self,
        name: str = "dasheng-base",
        loss: str = "BCELoss",
        **kwargs,
    ):
        r"""
        Configuration class for the Dasheng model.

        Args:
            name (str, *optional*):
                Can be "dasheng-base", "dasheng-0.6B", or "dasheng-1.2B". Default to "dasheng-base".
            loss (str, *optional*):
                Name of the loss function to use. Can be any loss in `nn.modules.loss`. Default to "BCELoss".
            kwargs (dict, *optional*):
                Additional keyword arguments, see `dasheng_model.modeling_dasheng.DashengFeatureExtractor` and `dasheng_model.modeling_dasheng.AudioTransformerMAE_Encoder` for more details.
        """

        super().__init__(**kwargs)

        encoder_kwargs = dict(target_length=1008, patch_size=[64, 4], patch_stride=[64, 4])

        if name == "dasheng-1.2B":
            encoder_kwargs["embed_dim"] = 1536
            encoder_kwargs["depth"] = 40
            encoder_kwargs["num_heads"] = 24
        elif name == "dasheng-0.6B":
            encoder_kwargs["embed_dim"] = 1280
            encoder_kwargs["depth"] = 32
            encoder_kwargs["num_heads"] = 16
        elif name == "dasheng-base":
            encoder_kwargs["embed_dim"] = 768
            encoder_kwargs["depth"] = 12
            encoder_kwargs["num_heads"] = 12
        else:
            raise ValueError(f"Unrecognized model name: {name}")
        self.name = name

        encoder_kwargs.update((k, kwargs[k]) for k in set(kwargs).intersection(encoder_kwargs))
        self.encoder_kwargs = {**encoder_kwargs, **kwargs}

        self.loss = loss
