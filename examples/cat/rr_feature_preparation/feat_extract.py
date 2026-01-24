import argparse
import os
import argparse
import numpy as np
import torch
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from models import ASTModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/torchhome'

class ASTFeatureExtractor(Dataset):
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

        else: # single audio
            self.fnames.append(self.audio_path) # no sizes
    
        checkpoint_path = self.model_path
        ast_mdl = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128,
                                  input_tdim=1024, imagenet_pretrain=False,
                                  audioset_pretrain=False, model_size='base384')
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        self.audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
        self.audio_model.load_state_dict(checkpoint)
        self.audio_model = self.audio_model.module
        self.audio_model = self.audio_model.to(torch.device("cuda"))

        for _, param in self.audio_model.named_parameters():
            param.requires_grad = False
        self.audio_model.eval()

        # other superparameters 
        self.target_length = 1024
        self.mel_bins = 128

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        wav_name = self.fnames[idx]
        waveform, sr = torchaudio.load(wav_name)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0,
            frame_shift=10)

        n_frames = fbank.shape[0]

        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank, wav_name
    
    def collate_fn(self, batch):
        feats = torch.stack([item[0] for item in batch], dim=0)
        wav_names = [item[1] for item in batch]
        return feats, wav_names
    
    def extract_features(self, feats):
        self.audio_model.eval()
        with torch.no_grad():
            output, features = self.audio_model.forward(feats, return_features=True)
        return features

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser:'
                                                 'python inference --audio_path ./0OxlgIitVig.wav '
                                                 '--model_path ./pretrained_models/audioset_10_10_0.4593.pth')

    # parser.add_argument("--model_path", type=str,
    #                     default='/opt/gpfs/home/chushu/exp/eat/ast_1_AS20k/ast_origin_implement/test-balanced-f10-t10-pTrue-b12-lr5e-5-decoupe/models/audio_model_wa.pth',
    #                     help="the trained model you want to test")
    parser.add_argument("--model_path", type=str,
                        default='/inspire/hdd/global_user/zhouchushu-253108120180/hubs/models/others/ast/audioset_10_10_0.4593.pth',
                        help="the trained model you want to test")
    parser.add_argument('--audio_path',
                        help='the audio you want to predict, sample rate 16k.',
                        default='/inspire/hdd/global_user/zhouchushu-253108120180/data/audioset/16k_wav_tsv/unbal_train.tsv',
                        type=str)
    parser.add_argument('--output_path',
                        help='the output path',
                        default='/inspire/hdd/global_user/zhouchushu-253108120180/data/features/ast_AS2M_pretrained_features',
                        type=str)

    args = parser.parse_args()
    
    batch_size = 48
    feature_extractor = ASTFeatureExtractor(args.audio_path, args.model_path)
    print(len(feature_extractor))
    feature_loader = DataLoader(feature_extractor, batch_size=batch_size, shuffle=False, num_workers=32, drop_last=False, collate_fn=feature_extractor.collate_fn, pin_memory=True)
    error_count = 0
    for i, (feats, wav_names) in tqdm(enumerate(feature_loader), total=len(feature_extractor)/batch_size):
        features = feature_extractor.extract_features(feats.to(torch.device("cuda")))
        # feats_ = []
        # features_ = []
        for key, value in features.items():
            for j, wav_name in enumerate(wav_names):
                os.makedirs(os.path.join(args.output_path, key), exist_ok=True)
                np.save(os.path.join(args.output_path, key, wav_name.split('/')[-1].replace('.wav', '.npy')), value[j].cpu().numpy())
        #     feat = make_features(wav_name, mel_bins=128)
        #     input_tdim = feat.shape[0]
        #     feat = feat.expand(1, input_tdim, 128)  
        #     feat = feat.to(torch.device("cuda"))
        #     feats_.append(feat.squeeze(0))
        #     with torch.no_grad():
        #         feature = feature_extractor.extract_features(feat)
        #     features_.append(feature.squeeze(0))
        # feats_ = torch.stack(feats_, dim=0)
        # features_ = torch.stack(features_, dim=0)
        # print(feats_.cpu().dtype, feats.cpu().dtype)
        # print(features.cpu().dtype, features_.cpu().dtype)
        # try:
        #     assert torch.allclose(feats_.cpu(), feats.cpu(), atol=1e-4), f'{feats_} {feats}'
        #     assert torch.allclose(features, features_, atol=1e-2), f'{features} {features_}'
        # except Exception as e:
        #     print(e)
        #     print(feats_.cpu().shape, feats.cpu().shape)
        #     print(features.cpu().shape, features_.cpu().shape)
        #     print(feats_.cpu().dtype, feats.cpu().dtype)
        #     print(features.cpu().dtype, features_.cpu().dtype)
        #     error_count += 1
        #     print(f'error_count: {error_count}')