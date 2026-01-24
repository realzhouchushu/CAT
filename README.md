# CAT

Representation-Regularized Convolutional Audio Transformer for Audio Understanding

## Requirements and Installation

Pip version 24.0 is required for the fairseq repository.
```shell
git clone git@github.com:realzhouchushu/CAT.git
cd CAT
pip install --editable ./
pip install -r EAT/requirements.txt
```

## Pre-train Recipe

### Data Preparation

#### 1 .Download datasets

**Audioset**: The main dataset employed in our experimental evaluations is [AudioSet](https://research.google.com/audioset/), which is publicly accessible on [Hugging Face](https://huggingface.co/datasets/confit/audioset-qiuqiangkong) at present. This [script](examples/cat/data_preparation/audioset_download.sh) audioset_download.sh may be helpful.

**Speech command v2**: SPC-v2 is available on [Kaggle](https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2).

**ESC-50**: ESC-50 is publicly accessible with this [link](https://github.com/karoldvl/ESC-50/archive/master.zip)


#### 2. Normalize your data
You should first resample the WAV files to the corresponding sample rates: **16 kHz** for AudioSet and Speech Commands v2, and **22 kHz** for ESC-50. Resampling the data online during training is time-consuming.

The [script](examples/cat/data_preparation/process/run_normalize.sh) can be used with slight modifications.

#### 3. Prepare your own data manifest

For pre-training without representation regularization, a raw .tsv file is sufficient, as demonstrated in this [folder](examples/cat/data_manifest/PRETRAIN_AS2M).

For supervised fine-tuning (SFT) on different datasets, you may need different setting folders, such as those for fine-tuning on [AS2M](examples/cat/data_manifest/SFT_AS2M), [AS20k](examples/cat/data_manifest/SFT_AS20k), [ESC-50](examples/cat/data_manifest/SFT_ESC_50), and [SPC-v2](examples/cat/data_manifest/SFT_SPC_2). These manifests can be used directly by only changing the first line to your own normalized data directory path.

In most cases, you can simply replace the first line of all .tsv files with the path to your own local data directory.

## Acknowledgements

The core architecture is adapted from [fairseq](https://github.com/facebookresearch/fairseq).

For the implementation of CAT, [EAT](https://github.com/cwx-worst-one/EAT) provided invaluable references for our work. Official repositories of [AST](https://github.com/YuanGongND/ast), [Audio-MAE](https://github.com/facebookresearch/AudioMAE) and [CLAP](https://github.com/LAION-AI/CLAP) are all integrated into this repository for feature extraction.

The code for audio captioning is adapted from [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM), and we utilize [aac-metrics](https://github.com/Labbeti/aac-metrics) to evaluate the quality of generated captions.