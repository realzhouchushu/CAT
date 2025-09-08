from evar.ar_base import (BaseAudioRepr, calculate_norm_stats)
from dataclasses import dataclass
import torch
import torchaudio
from enum import Enum, auto
from functools import partial
import torch.nn.functional as F

# EAT utilize cls token for prediction in most downstream tasks
class PredictionMode(Enum):
    MEAN_POOLING = auto()
    CLS_TOKEN = auto()
    LIN_SOFTMAX = auto()
    # NOTE: (chushu) [2025.08.29]: tmp useless
    CLAP_TYPE1 = auto()
    CLAP_TYPE2 = auto()
    CLAP_TYPE3 = auto()
    CLAP_TYPE4 = auto()
    CLAP_TYPE5 = auto()
    CLAP_TYPE6 = auto()

class EAT_Feature(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, waveforms): # dataset默认单通道wav
        def get_one(waveform):
            assert waveform.dim() == 1, "Only mono-channel audio is supported"
            waveform, sr = waveform.cuda(), self.cfg.sample_rate
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
                print(f"Resampled to 16kHz: {waveform.shape}")

            # Normalize and convert to mel-spectrogram
            waveform = waveform - waveform.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            ).unsqueeze(0)

            # Pad or truncate to target length
            n_frames = mel.shape[1]
            if n_frames < self.cfg.target_length:
                mel = torch.nn.ZeroPad2d((0, 0, 0, self.cfg.target_length - n_frames))(mel)
            elif n_frames > self.cfg.target_length:
                mel = mel[:, :self.cfg.target_length, :]

            mel = mel.cuda()  # shape: [1, T, F]

            return mel
    
        device = waveforms.device
        if len(waveforms.shape) == 1:  # [L] -> [1, L]
            waveforms = waveforms.unsqueeze(0)
        fbanks = torch.stack([get_one(w) for w in waveforms])
        return fbanks.to(device)

class AR_EAT(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.to_feature = EAT_Feature(cfg)
        self.model = self.load_model()

    # TODO: (chushu) [2025.09.03] 检查计算均值方差是否必要
    def precompute(self, device, data_loader):
        if self.cfg.get('norm_mean', None) is not None and self.cfg.get('norm_std', None) is not None:
            self.norm_stats = [self.cfg.norm_mean, self.cfg.norm_std]
        else:
            self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def load_model(self):
        @dataclass
        class UserDirModule:
            user_dir: str

        if self.cfg.framework == "huggingface":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(self.cfg.weight_file, trust_remote_code=True).eval().cuda()
        elif self.cfg.framework == "fairseq":
            import fairseq
            model_path = UserDirModule(self.cfg.model_dir)
            fairseq.utils.import_user_module(model_path)
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.cfg.weight_file])
            model = models[0]
            # print(model)
            # if self.cfg.mode == "finetune":
            #     model = model.model
            model.eval().cuda()
        else:
            raise ValueError(f"Unsupported framework: {self.cfg.framework}")
        return model
    
    def extract_feature_tensor(self, x):
        if self.cfg.framework == "huggingface":
            return self.model.extract_features(x)
        elif self.cfg.framework == "fairseq":
            return self.model.extract_features(
                x,
                mode="IMAGE",
                mask=False,
                remove_extra_tokens=(
                    self.cfg.prediction_mode != PredictionMode.CLS_TOKEN
                ),
            )["x"]
        else:
            raise ValueError(f"Unsupported framework: {self.cfg.framework}")
    
    def forward(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = self.normalize_spectrogram(x)
        x = self.augment_if_training(x)

        x = self.extract_feature_tensor(x)
        # print(x.shape)

        # different prediction mode
        prediction_mode = PredictionMode[self.cfg.prediction_mode] if isinstance(self.cfg.prediction_mode, str) else self.cfg.prediction_mode
        if prediction_mode == PredictionMode.MEAN_POOLING:
            x = x.mean(dim=1)
        elif prediction_mode == PredictionMode.CLS_TOKEN:
            x = x[:, 0]
        elif prediction_mode == PredictionMode.LIN_SOFTMAX:
            dtype = x.dtype
            x = F.logsigmoid(x.float())
            x = torch.logsumexp(x + x, dim=1) - torch.logsumexp(x + 1e-6, dim=1)
            x = x.clamp(max=0)
            x = x - torch.log(-(torch.expm1(x)))
            x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
            x = x.to(dtype=dtype)
        elif prediction_mode == PredictionMode.CLAP_TYPE1 or prediction_mode == PredictionMode.CLAP_TYPE3 or prediction_mode == PredictionMode.CLAP_TYPE5:
            raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")
        elif prediction_mode == PredictionMode.CLAP_TYPE2 or prediction_mode == PredictionMode.CLAP_TYPE4 or prediction_mode == PredictionMode.CLAP_TYPE6:
            x = x[:,0]
            x = self.model.clap_proj(x)
        else:
            raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")
        return x
    
    def normalize_spectrogram(self, spectrograms):
        mu, sigma = self.norm_stats
        spectrograms = (spectrograms - mu) / (sigma * 2) # follows the original AudiosetDataset
        return spectrograms