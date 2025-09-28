import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

# ===== Argument Parser =====
def get_parser(): # !!!Modify
    parser = argparse.ArgumentParser(description="Extract EAT features for downstream tasks")
    parser.add_argument('--source_file', help='Path to input .wav file', default='/opt/gpfs/home/chushu/data/audioset/16k_wav_tsv/eval.tsv') # !!!Modify
    parser.add_argument('--target_file', help='Path to output .npy file', default='/opt/gpfs/home/chushu/data/features/eat_clap_feature')
    parser.add_argument('--model_dir', help='Directory containing the model definition (not needed for HF framework)', default='/opt/gpfs/home/chushu/codes/2506/EAT/EAT')
    parser.add_argument('--checkpoint_dir', help='Checkpoint path or HF model ID', default='/opt/gpfs/home/chushu/exp/eat/pre_4_AS2M/clap_0_2025-08-27_09-23-59/checkpoint_last.pt')
    parser.add_argument('--granularity', choices=['all', 'frame', 'utterance'],
                        help='Feature type: "all" (including CLS), "frame" (excluding CLS), or "utterance" (CLS only)', default='utterance')
    parser.add_argument('--target_length', type=int, help='Target length of mel-spectrogram', default=1024)
    parser.add_argument('--norm_mean', type=float, default=-4.268, help='Normalization mean')
    parser.add_argument('--norm_std', type=float, default=4.569, help='Normalization std')
    parser.add_argument('--mode', choices=['pretrain', 'finetune'], help='Model mode', default='pretrain')
    parser.add_argument('--framework', choices=['fairseq', 'huggingface'], help='Framework to use', default='fairseq')
    return parser

# # ===== Feature Extraction For Two Frameworks =====
# def extract_feature_tensor(model, x, framework):
#     if framework == "huggingface":
#         return model.extract_features(x)
#     elif framework == "fairseq":
#         return model.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=False)['x']
#     else:
#         raise ValueError(f"Unsupported framework: {framework}")


# # ===== Feature Extraction Pipeline =====
# def extract_features(args):
#     assert args.source_file.endswith('.wav'), "Source file must be a .wav file"

#     # Load waveform and resample to 16kHz if necessary
#     wav, sr = sf.read(args.source_file)
#     assert sf.info(args.source_file).channels == 1, "Only mono-channel audio is supported"
#     waveform = torch.tensor(wav).float().cuda()
#     if sr != 16000:
#         waveform = torchaudio.functional.resample(waveform, sr, 16000)
#         print(f"Resampled to 16kHz: {args.source_file}")

#     # Normalize and convert to mel-spectrogram
#     waveform = waveform - waveform.mean()
#     mel = torchaudio.compliance.kaldi.fbank(
#         waveform.unsqueeze(0),
#         htk_compat=True,
#         sample_frequency=16000,
#         use_energy=False,
#         window_type='hanning',
#         num_mel_bins=128,
#         dither=0.0,
#         frame_shift=10
#     ).unsqueeze(0)

#     # Pad or truncate to target length
#     n_frames = mel.shape[1]
#     if n_frames < args.target_length:
#         mel = torch.nn.ZeroPad2d((0, 0, 0, args.target_length - n_frames))(mel)
#     elif n_frames > args.target_length:
#         mel = mel[:, :args.target_length, :]

#     mel = (mel - args.norm_mean) / (args.norm_std * 2)
#     mel = mel.unsqueeze(0).cuda()  # shape: [1, 1, T, F]

#     model = load_model(args)

#     with torch.no_grad():
#         try:
#             result = extract_feature_tensor(model, mel, args.framework)

#             if args.granularity == 'frame':
#                 result = result[:, 1:, :]     # remove CLS token
#             elif args.granularity == 'utterance':
#                 result = result[:, 0]         # keep only CLS token

#             result = result.squeeze(0).cpu().numpy()
#             np.save(args.target_file, result)
#             print(f"Feature shape: {result.shape}")
#             print(f"Saved to: {args.target_file}")

#         except Exception as e:
#             print(f"Feature extraction failed: {e}")
#             raise

@dataclass
class UserDirModule:
    user_dir: str

class EATFeatureExtractor(Dataset):
    def __init__(self, args):
        self.args = args
        self.audio_path = args.source_file
        self.fnames = []
        self.sizes = []
        print(self.audio_path)
        if self.audio_path.endswith(".tsv"):
            with open(self.audio_path, "r") as f:
                root_dir = f.readline().strip()
                for i, line in enumerate(f):
                    items = line.strip().split()
                    assert len(items) == 2, line
                    sz = int(items[1])
                    self.fnames.append(os.path.join(root_dir, items[0]))
                    self.sizes.append(sz)

        else: # single audio
            self.fnames.append(self.audio_path) # no sizes
        
        self.model = self.load_model()
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        wav_name = self.fnames[idx]
        wav, sr = sf.read(wav_name)
        assert sf.info(wav_name).channels == 1, "Only mono-channel audio is supported"
        waveform = torch.tensor(wav).float()
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            print(f"Resampled to 16kHz: {wav_name}")

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
        if n_frames < args.target_length:
            mel = torch.nn.ZeroPad2d((0, 0, 0, args.target_length - n_frames))(mel)
        elif n_frames > args.target_length:
            mel = mel[:, :args.target_length, :]

        mel = (mel - args.norm_mean) / (args.norm_std * 2)
        mel = mel # shape: [1, T, F]


        return mel, wav_name
    
    def collate_fn(self, batch):
        feats = torch.stack([item[0] for item in batch], dim=0)
        wav_names = [item[1] for item in batch]
        return feats, wav_names
    
    def extract_features(self, x):
        with torch.no_grad():
            try:
                if self.args.framework == "huggingface":
                    return self.model.extract_features(x)
                elif self.args.framework == "fairseq":
                    return self.model.extract_features(x, mode="IMAGE", mask=False, remove_extra_tokens=False)
                else:
                    raise ValueError(f"Unsupported framework: {self.args.framework}")
            except Exception as e:
                print(f"Feature extraction failed: {e}")
                raise
            

    # ===== Model Loader =====
    def load_model(self):
        if self.args.framework == "huggingface":
            from transformers import AutoModel
            model = AutoModel.from_pretrained(self.args.checkpoint_dir, trust_remote_code=True).eval().cuda()
        elif self.args.framework == "fairseq":
            import fairseq
            model_path = UserDirModule(self.args.model_dir)
            fairseq.utils.import_user_module(model_path)
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.args.checkpoint_dir])
            model = models[0]
            if self.args.mode == "finetune":
                model = model.model
            model.eval().cuda()
        else:
            raise ValueError(f"Unsupported framework: {self.args.framework}")
        return model

# ===== Entry =====
if __name__ == '__main__':
    args = get_parser().parse_args()
    batch_size = 128
    feature_extractor = EATFeatureExtractor(args)
    feature_loader = DataLoader(feature_extractor, batch_size=batch_size, shuffle=False, num_workers=48, drop_last=False, collate_fn=feature_extractor.collate_fn)
    error_count = 0
    if 'eval' in args.source_file:
        feature_extractor.model.eval() # !!!Modify
    for i, (feats, wav_names) in tqdm(enumerate(feature_loader), total=len(feature_extractor)/batch_size): 
        features = feature_extractor.extract_features(feats.to(torch.device("cuda")))
        # feats_ = []
        # features_ = []
        # for key, value in features.items():
        for layer_idx in range(len(features['layer_results']) + 1):
            if layer_idx != 0 and 'eval' in args.source_file: # !!!Modify
                continue
            if layer_idx == 0 and 'eval' not in args.source_file: # !!!Modify
                for j, wav_name in enumerate(wav_names):
                    os.makedirs(os.path.join(args.target_file, str('train_mel')), exist_ok=True)
                    np.save(os.path.join(args.target_file, str('train_mel'), wav_name.split('/')[-1].replace('.wav', '.npy')), feats[j].detach().cpu().numpy())
            value = features['layer_results'][layer_idx-1] if layer_idx > 0 else features['x']
            if 'eval' in args.source_file:
                layer_idx = "eval" # !!!Modify
            for j, wav_name in enumerate(wav_names):
                os.makedirs(os.path.join(args.target_file, str(layer_idx)), exist_ok=True)
                np.save(os.path.join(args.target_file, str(layer_idx), wav_name.split('/')[-1].replace('.wav', '.npy')), value[j][0].detach().cpu().numpy())
            # eval set mel save scrpits
            if layer_idx == 'eval':
                layer_idx = 'eval_mel'
                for j, wav_name in enumerate(wav_names):
                    os.makedirs(os.path.join(args.target_file, str(layer_idx)), exist_ok=True)
                    np.save(os.path.join(args.target_file, str(layer_idx), wav_name.split('/')[-1].replace('.wav', '.npy')), feats[j].detach().cpu().numpy())