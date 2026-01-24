---
license: apache-2.0
tags:
- audio
- audio-classification
---

# Dasheng: a large scale general-purpose audio encoder

Dasheng (**D**eep **A**udio-**S**ignal **H**olistic **E**mbeddi**ng**s), or “大声” ("great sound"), is a general-purpose audio encoder trained on a large-scale self-supervised learning task. Dasheng is designed to capture rich audio information across various domains, including speech, music, and environmental sounds. The model is trained on 272,356 hours of diverse audio data with 1.2 billion parameters, and exhibits significant performance gains on the [HEAR benchmark](https://hearbenchmark.com/). Dasheng outperforms previous works on CREMA-D, LibriCount, Speech Commands, VoxLingua, and competes well in music and environmental sound classification tasks.

**Original Repository:** [https://github.com/RicherMans/Dasheng](https://github.com/RicherMans/Dasheng)

![dasheng](https://raw.githubusercontent.com/jimbozhang/hf_transformers_custom_model_dasheng/main/pic/hear_eval.png)

## Usage

### Inference

```python
>>> model_name = "mispeech/dasheng-base"

>>> from transformers import AutoModel, AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
>>> model = AutoModel.from_pretrained(model_name, outputdim=None, trust_remote_code=True)

>>> import torch
>>> inputs = feature_extractor(torch.randn(1, 16000), sampling_rate=sampling_rate, return_tensors="pt")
>>> inputs.input_values.shape
torch.Size([1, 64, 101])   # 64 mel-filterbanks, 101 frames

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> outputs.hidden_states.shape
torch.Size([1, 25, 768])   # 25 T-F patches (patch size 64x4, no overlap), before mean-pooling

>>> outputs.logits.shape
torch.Size([1, 768])   # mean-pooled embedding (would be logits from a linear layer if `outputdim` was set)
```

### Fine-tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimbozhang/hf_transformers_custom_model_dasheng/blob/main/example_finetune_esc50.ipynb)

[`example_finetune_esc50.ipynb`](https://github.com/jimbozhang/hf_transformers_custom_model_dasheng/blob/main/example_finetune_esc50.ipynb) demonstrates how to train a linear head on the ESC-50 dataset with the Dasheng encoder frozen.

## Citation

If you find Dasheng useful in your research, please consider citing the following paper:

```bibtex
@inproceedings{dinkel2023scaling,
  title={Scaling up masked audio encoder learning for general audio classification},
  author={Dinkel, Heinrich and Yan, Zhiyong and Wang, Yongqing and Zhang, Junbo and Wang, Yujun and Wang, Bin},
  booktitle={Interspeech 2024},
  year={2024}
}
```
