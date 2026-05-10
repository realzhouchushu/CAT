# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .pretraining_AS2M import MaeImagePretrainingTask, MaeImagePretrainingConfig
from .finetuning import MaeImageClassificationTask, MaeImageClassificationConfig


__all__ = [
    "MaeImagePretrainingTask",
    "MaeImagePretrainingConfig",
    "MaeImageClassificationTask",
    "MaeImageClassificationConfig",
]