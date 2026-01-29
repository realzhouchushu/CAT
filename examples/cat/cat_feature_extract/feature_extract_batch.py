import argparse
import os
import sys
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp


# =====================
#   Model Loader
# =====================
def load_model(args):
    if args.framework == "huggingface":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.checkpoint_dir, trust_remote_code=True).eval().cuda()
    elif args.framework == "fairseq":
        import fairseq
        fairseq.utils.import_user_module(args.model_dir)

        eat_root = "~/codes/2506/EAT"
        if eat_root not in sys.path:
            sys.path.append(eat_root)

        from EAT.tasks import finetuning
        from EAT.models import MaeImageClassificationModel, Data2VecMultiModel

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.checkpoint_dir])
        model = models[0]
        if args.mode == "finetune":
            model = model.model
        model.eval().cuda()
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")
    return model


def extract_feature_tensor(model, x, framework):
    if framework == "huggingface":
        return model.extract_features(x)
    elif framework == "fairseq":
        return model.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=False)['x']
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# =====================
#   Dataset
# =====================
class EATFeatureExtractor(Dataset):
    def __init__(self, audio_path, args):
        self.audio_path = audio_path
        self.args = args
        self.target_length = args.target_length
        self.mel_bins = 128
        self.norm_mean = args.norm_mean
        self.norm_std = args.norm_std

        self.fnames = []
        self.sizes = []
        print(f"Loading audio list from: {audio_path}")
        if audio_path.endswith(".tsv"):
            with open(audio_path, "r") as f:
                root_dir = f.readline().strip()
                for line in f:
                    items = line.strip().split()
                    if len(items) == 2:
                        self.fnames.append(os.path.join(root_dir, items[0]))
                        self.sizes.append(int(items[1]))
        else:
            self.fnames.append(audio_path)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        wav_name = self.fnames[idx]
        waveform, sr = torchaudio.load(wav_name)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[:self.target_length, :]

        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        fbank = fbank.unsqueeze(0)
        return fbank, wav_name

    def collate_fn(self, batch):
        feats = torch.stack([item[0] for item in batch], dim=0)
        wav_names = [item[1] for item in batch]
        return feats, wav_names


# =====================
#   DDP Worker
# =====================
def main_worker(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"🚀 Using {world_size} GPUs for feature extraction...")

    model = load_model(args)
    model = model.cuda(rank)
    model.eval()

    dataset = EATFeatureExtractor(args.audio_path, args)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers // max(1, world_size),
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    for feats, wav_names in tqdm(dataloader, disable=(rank != 0)):
        feats = feats.cuda(rank, non_blocking=True)
        with torch.no_grad():
            result = extract_feature_tensor(model, feats, args.framework)
            if args.granularity == 'frame':
                result = result[:, 1:, :]
            elif args.granularity == 'utterance':
                result = result[:, 0, :]
            result = result.cpu().numpy()

            for i, wav_name in enumerate(wav_names):
                out_path = os.path.join(
                    args.output_path,
                    os.path.basename(wav_name).replace('.wav', f'_gpu{rank}.npy')
                )
                np.save(out_path, result[i])

    dist.barrier()
    if rank == 0:
        print(f"✅ All GPUs finished! Features saved to: {args.output_path}")

    dist.destroy_process_group()


# =====================
#   Main
# =====================
def main():
    parser = argparse.ArgumentParser(description='Extract EAT features for a dataset')

    # ===== 默认参数保留 =====
    parser.add_argument('--audio_path',
                        default='~/data/audioset/16k_wav_tsv/unbal_train.tsv',
                        type=str)
    parser.add_argument('--output_path',
                        default='~/data/features/eat_features',
                        type=str)
    parser.add_argument('--model_dir',
                        default='~/codes/2506/EAT/EAT',
                        type=str)
    parser.add_argument('--checkpoint_dir',
                        default='~/exp/eat/sft_4_AS2M/default_lw1_llayer0_layer12_llayer0/default_0_2025-09-20_15-33-21/checkpoint_best.pt',
                        type=str)
    parser.add_argument('--framework', default='fairseq', choices=['fairseq', 'huggingface'])
    parser.add_argument('--mode', default='finetune', choices=['pretrain', 'finetune'])
    parser.add_argument('--granularity', default='utterance', choices=['all', 'frame', 'utterance'])
    parser.add_argument('--target_length', default=1024, type=int)
    parser.add_argument('--norm_mean', default=-4.268, type=float)
    parser.add_argument('--norm_std', default=4.569, type=float)
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--num_workers', default=96, type=int)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


if __name__ == '__main__':
    main()
