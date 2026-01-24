import argparse
import os
import argparse
import numpy as np
import torch
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# from models import beats
from beats.BEATs import BEATs, BEATsConfig

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/torchhome'

class BEATsFeatureExtractor(Dataset):
    def __init__(self, audio_path, model_path):
        self.audio_path = audio_path
        self.model_path = model_path
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
        elif self.audio_path.endswith(".scp"):
            with open(self.audio_path, "r") as f:
                for i, line in enumerate(f):
                    items = line.strip()
                    self.fnames.append(items)
                    # self.sizes.append(sz)
        else: # single audio
            self.fnames.append(self.audio_path) # no sizes
    
        checkpoint_path = self.model_path
        checkpoint = torch.load(checkpoint_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.audio_model = BEATs(cfg)
        self.audio_model.load_state_dict(checkpoint['model'])
        self.audio_model.to(torch.device("cuda"))

        for _, param in self.audio_model.named_parameters():
            param.requires_grad = False
        self.audio_model.eval()

        # other superparameters 
        self.target_length = 1024
        self.mel_bins = 128

    def __len__(self):
        return len(self.fnames)

    def make_features(self, wav_name):
        waveform, sr = torchaudio.load(wav_name)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[-1] < 160000:
            waveform = torch.cat([waveform, torch.zeros(1, 160000 - waveform.shape[-1])], dim=-1)
        elif waveform.shape[-1] > 160000:
            waveform = waveform[:, :160000]
        waveform = waveform[0]
        return waveform

    def __getitem__(self, idx):
        wav_name = self.fnames[idx]
        # waveform, sr = torchaudio.load(wav_name)
        waveform = self.make_features(wav_name)
        return waveform, wav_name
    
    def collate_fn(self, batch):
        feats = torch.stack([item[0] for item in batch], dim=0)
        wav_names = [item[1] for item in batch]
        return feats, wav_names
    
    def extract_features(self, feats):
        self.audio_model.eval()
        padding_mask = feats.new_zeros(feats.shape[0], feats.shape[1]).bool()
        with torch.no_grad():
            representation = self.audio_model.extract_features(feats, padding_mask=padding_mask)[0]
        return representation



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    # parser.add_argument("--model_path", type=str,
    #                     default='/opt/gpfs/home/chushu/exp/eat/ast_1_AS20k/ast_origin_implement/test-balanced-f10-t10-pTrue-b12-lr5e-5-decoupe/models/audio_model_wa.pth',
    #                     help="the trained model you want to test")
    parser.add_argument("--model_path", type=str,
                        default='./pretrained_models/BEATs_iter3_plus_AS2M.pt',
                        # default='./pretrained_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
                        help="the trained model you want to test")
    parser.add_argument('--audio_path',
                        help='the audio you want to predict, sample rate 16k.',
                        # default='/inspire/hdd/global_user/zhouchushu-253108120180/data/audioset/16k_wav_tsv/unbal_train.tsv',
                        default='/inspire/hdd/global_user/zhouchushu-253108120180/data/audioset/16k_wav_tsv/unbal_train.tsv',
                        type=str)
    parser.add_argument('--output_path',
                        help='the output path',
                        # default='/inspire/hdd/global_user/zhouchushu-253108120180/data/features/ast_AS2M_pretrained_features',
                        default="/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/features/audioset/beats_pretrain_features",
                        type=str)

    args = parser.parse_args()
    
    batch_size = 64
    feature_extractor = BEATsFeatureExtractor(args.audio_path, args.model_path)
    print(len(feature_extractor))
    feature_loader = DataLoader(feature_extractor, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, collate_fn=feature_extractor.collate_fn, pin_memory=True)
    error_count = 0
    for i, (feats, wav_names) in tqdm(enumerate(feature_loader), total=len(feature_extractor)/batch_size):
        features = feature_extractor.extract_features(feats.to(torch.device("cuda")))
        features = {"sft": features.mean(dim=1)}
        # import pdb ; pdb.set_trace()
        for key, value in features.items():
            for j, wav_name in enumerate(wav_names):
                os.makedirs(os.path.join(args.output_path, key), exist_ok=True)
                # print(value[j].cpu().numpy().shape)
                np.save(os.path.join(args.output_path, key, wav_name.split('/')[-1].replace('.wav', '.npy')), value[j].cpu().numpy())