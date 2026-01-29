import random
import json
import os
import argparse
import csv
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio


import timm
from timm.models.layers import to_2tuple
from timm.models.layers import trunc_normal_

import models_vit

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.reader(f)  # 使用 reader 而不是 DictReader（因为没有 header）
        line_count = 0
        for row in csv_reader:
            if len(row) >= 2:  # 确保至少有 index 和 mid 两列
                line_count += 1
                index = row[0]  # 第一列是 index
                mid = row[1]    # 第二列是 mid
                index_lookup[mid] = index
    return index_lookup

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        if dataset_json_file.endswith(".json"):
            with open(dataset_json_file, 'r') as fp:
                data_json = json.load(fp)
            self.data = data_json['data']
        elif dataset_json_file.endswith(".tsv"):
            self.raw_labels = {}
            self.data = []
            with open(dataset_json_file.replace(".tsv", ".lbl"), "r") as f:
                for line in f.readlines():
                    items = line.strip().split()
                    self.raw_labels[items[0]] = items[1]
            with open(dataset_json_file, "r") as f:
                root_dir = f.readline().strip()
                for i, line in enumerate(f):
                    current_data = {}
                    items = line.strip().split()
                    assert len(items) == 2, line
                    sz = int(items[1])
                    current_data['wav'] = os.path.join(root_dir, items[0])
                    current_data['labels'] = self.raw_labels[items[0].split('.')[0]]
                    self.data.append(current_data)
        
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')


    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        # 512
        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda


    def _fbank(self, filename, filename2=None):
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)
            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)
            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'], mix_datum['wav'])
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]
            label_indices = np.zeros(self.label_num)
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'])
            else:
                fbank, mix_lambda = self._fbank(datum['wav'])
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.index_dict[label_str])
        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        if self.noise == True: # default is false, true for spc
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), label_indices, datum['wav']

    def __len__(self):
        return len(self.data)

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def get_args_parser():
    # base fields from AudioMAE codebase
    parser = argparse.ArgumentParser('AudioMAE test', add_help=False)
    parser.add_argument('--model', type=str, default='vit_base_patch16')
    parser.add_argument('--mixup', type=float, default=0.5, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--nb_classes', type=int, default=527)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--mask_2d', type=bool, default=True)
    parser.add_argument('--use_custom_patch', type=bool, default=False)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--finetune', default='~/hubs/models/others/audio-mae/pretrained.pth', help='finetune from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only') # XXX: other field can use default value while this field is not.
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "k400"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str, default="/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank", help="fbank dir")
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument("--label_csv", type=str, default='~/data/audioset/label_descriptors.csv', help="csv with class labels")
    parser.add_argument("--data_train", type=str, default='~/data/audioset/16k_wav_tsv/unbal_train.tsv', help="training data json")
    parser.add_argument("--data_eval", type=str, default='~/data/audioset/16k_wav_tsv/eval.tsv', help="validation data json")
    # customized fields
    parser.add_argument('--output_path',
                        help='the output path',
                        default='~/data/features/audio_mae_pretrained_features',
                        type=str)
    return parser

def main(args):
    device = torch.device(args.device)
    norm_stats = {'audioset':[-4.2677393, 4.5689974], 'k400':[-4.2677393, 4.5689974], 
                    'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
    target_length = {'audioset':1024, 'k400':1024, 'esc50':512, 'speechcommands':128}
    multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
    audio_conf_train = {'num_mel_bins': 128, 
                    'target_length': target_length[args.dataset], 
                    'freqm': 48,
                    'timem': 192,
                    'mixup': args.mixup,
                    'dataset': args.dataset,
                    'mode':'train',
                    'mean':norm_stats[args.dataset][0],
                    'std':norm_stats[args.dataset][1],
                    'noise':False,
                    'multilabel':multilabel_dataset[args.dataset],
                    }
    audio_conf_val = {'num_mel_bins': 128, 
                    'target_length': target_length[args.dataset], 
                    'freqm': 0,
                    'timem': 0,
                    'mixup': 0,
                    'dataset': args.dataset,
                    'mode':'val',
                    'mean':norm_stats[args.dataset][0],
                    'std':norm_stats[args.dataset][1],
                    'noise':False,
                    'multilabel':multilabel_dataset[args.dataset],
                    }  
    # HACK: train set is used for feature extraction, so we use audio_conf_val to get the features
    dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf_val, 
                                    use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                    roll_mag_aug=False, load_video=args.load_video, mode='eval')
    dataset_val = AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=audio_conf_val, 
                                    use_fbank=args.use_fbank, fbank_dir=args.fbank_dir, 
                                    roll_mag_aug=False, load_video=args.load_video, mode='eval')
    print(f"="*100)
    print(f"number of training samples: {len(dataset_train)}")
    print(f"number of validation samples: {len(dataset_val)}")
    print(f"="*100)
    
    print(f"="*100)
    print(f"loading model architecture: {args.model}")
    print(f"="*100)
    x = dataset_train[0]
    print(f"x: {x}")

    batch_size = 64
    feature_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        mask_2d=args.mask_2d,
        use_custom_patch=args.use_custom_patch,
        ## remove video part for A-MAE
        #load_video=args.load_video,
        # n_frm=args.n_frm,
        #split_pos=args.split_pos,
        #av_fusion=args.av_fusion,
    )
    img_size = (1024, 128)
    in_chans = 1
    emb_dim = 768
    if args.model == "vit_small_patch16":
        emb_dim = 384
    if args.use_custom_patch:
        model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=16, in_chans=1, embed_dim=emb_dim, stride=10)
        model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding
    else:
        model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
        num_patches = model.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

    model.to(device)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        if not args.eval:
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
        # load pre-trained model
        checkpoint_keys = set(checkpoint_model.keys())
        state_dict_keys = set(state_dict.keys())
        print(f"Keys in checkpoint_model but NOT in state_dict: {checkpoint_keys - state_dict_keys}")
        print(f"Keys in state_dict but NOT in checkpoint_model: {state_dict_keys - state_dict_keys}")
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        if not args.eval:
            trunc_normal_(model.head.weight, std=2e-5)
    
    model.to(device)
    model.eval()

    success_count = 0
    for i, (feats, targets, wav_names) in tqdm(enumerate(feature_loader), total=len(dataset_train)/batch_size):
        features = model.forward_features(feats.to(device))
        # print(f"features: {features}")
        # print(f"features.shape: {features.shape}")
        # # print(f"targets: {targets}")
        # print(f"wav_names: {wav_names}")
        # break
        for j, wav_name in enumerate(wav_names):
            os.makedirs(args.output_path, exist_ok=True)
            np.save(os.path.join(args.output_path, wav_name.split('/')[-1].replace('.wav', '.npy')), features[j].cpu().detach().numpy())
            success_count += 1
    print(f"success_count: {success_count}")
    print(f"total_count: {len(dataset_train)}")
    print(f"success_rate: {success_count/len(dataset_train)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('AudioMAE test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)