import argparse
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import csv
import os
from dataclasses import dataclass
from transformers import AutoModel
import numpy as np

# ===== Argument Parser =====
def get_parser():
    parser = argparse.ArgumentParser(description="Use fine-tuned CAT for acoustic event classification")
    parser.add_argument('--source_file', required=True, help='Path to input .wav or .m4a file')
    parser.add_argument('--label_file', required=True, help='Path to label CSV file')
    parser.add_argument('--model_dir', required=True, help='Model definition directory')
    parser.add_argument('--checkpoint_dir', required=True, help='Checkpoint path')
    parser.add_argument('--target_length', type=int, default=1024, help='Target mel length (time dimension)')
    parser.add_argument('--top_k_prediction', type=int, default=5, help='Top-k predicted labels')
    parser.add_argument('--norm_mean', type=float, default=-4.268, help='Normalization mean')
    parser.add_argument('--norm_std', type=float, default=4.569, help='Normalization std')
    parser.add_argument('--framework', required=True, choices=['fairseq', 'huggingface'], help='Model framework')
    parser.add_argument('--num_classes', type=int, default=527, help='Number of classes for AudioSet')
    parser.add_argument('--prediction_mode', type=str, default='CLS_TOKEN',
                        choices=['CLS_TOKEN', 'MEAN_POOLING', 'FEATURE'],
                        help='Prediction mode: CLS_TOKEN for classification, MEAN_POOLING for averaging, FEATURE for feature extraction')
    parser.add_argument('--audio_segment', type=str, default=None,
                        help='Audio segment to analyze in format "start_sec-end_sec" (e.g., "20-30" for 20s to 30s)')
    return parser


# ===== Label Loader =====
def build_dictionary(label_path):
    vocab = {}
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            index, label = int(row[0]), row[2]
            vocab[index] = label
    return vocab


# ===== Fairseq Helper =====
@dataclass
class UserDirModule:
    user_dir: str


# ===== Model Loader =====
def load_model(args):
    if args.framework == 'huggingface':
        model = AutoModel.from_pretrained(args.checkpoint_dir, trust_remote_code=True).eval().cuda()
    elif args.framework == 'fairseq':
        import fairseq
        model_path = UserDirModule(args.model_dir)
        fairseq.utils.import_user_module(model_path)

        # Load checkpoint with task info
        models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.checkpoint_dir])
        model = models[0]
        model = model.eval().cuda()

        # Check if model has classification head
        if hasattr(model, 'head'):
            print("Model has classification head")
        else:
            print("Model is pretraining model without classification head")
            print("Will use feature extraction + cosine similarity for inference")
    else:
        raise ValueError(f"Unsupported framework: {args.framework}")
    return model


# ===== Audio Preprocessing =====
def preprocess_audio(path, target_length, norm_mean, norm_std, audio_segment=None):
    # Support both .wav and .m4a files
    import librosa

    assert path.endswith('.wav') or path.endswith('.m4a'), "Input must be a .wav or .m4a file"

    # Load audio using librosa (supports more formats)
    wav, sr = librosa.load(path, sr=None, mono=True)

    # Handle audio segment
    if audio_segment is not None:
        start_sec, end_sec = map(float, audio_segment.split('-'))
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        wav = wav[start_sample:end_sample]
        print(f"Audio segment: {start_sec}s - {end_sec}s ({end_sec - start_sec}s)")

    wav = torch.tensor(wav).float()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav - wav.mean()  # normalization as in training

    # Convert to mel spectrogram (same as FileAudioDataset)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
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
    if n_frames < target_length:
        mel = torch.nn.ZeroPad2d((0, 0, 0, target_length - n_frames))(mel)
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    # Global normalization (same as in FileAudioDataset)
    mel = (mel - norm_mean) / (norm_std * 2)

    # Shape: [1, 1, T, F] = [1, 1, 1024, 128] - need 4D tensor for CNN encoder
    return mel.unsqueeze(0).cuda()  # Add channel dimension: [1, 1, 1024, 128]


# ===== Main Inference =====
def main():
    args = get_parser().parse_args()

    model = load_model(args)
    vocab = build_dictionary(args.label_file)
    mel = preprocess_audio(args.source_file, args.target_length, args.norm_mean, args.norm_std)

    print(f"Input mel shape: {mel.shape}")

    with torch.no_grad():
        try:
            # Check model type and use appropriate forward method
            if hasattr(model, 'head') and hasattr(model, 'model_forward'):
                # Classification model (EAT_audio_classification)
                print("Using classification model with head")
                x = model.model_forward(mel)

                # Apply prediction mode
                if args.prediction_mode == 'MEAN_POOLING':
                    x = x.mean(dim=1)
                elif args.prediction_mode == 'CLS_TOKEN':
                    x = x[:, 0]

                # Apply fc_norm if exists
                if hasattr(model, 'fc_norm') and model.fc_norm is not None:
                    x = model.fc_norm(x)

                # Apply classification head
                logits = model.head(x)
                probs = torch.sigmoid(logits)

                values, indices = torch.topk(probs, args.top_k_prediction)

                print("\n************ Acoustic Event Inference ************")
                print("LABEL".ljust(30) + "PREDICTION")
                for i, v in zip(indices[0], values[0]):
                    print(f"{vocab[i.item()]:<30} {v.item():.3f}")
                print("**************************************************\n")

            elif hasattr(model, 'extract_features'):
                # Pretraining model (EAT_pretraining) - use feature extraction
                print("Using pretraining model for feature extraction")
                model_outputs = model.extract_features(
                    mel,
                    mode="IMAGE",
                    mask=False,
                    remove_extra_tokens=False,
                )
                x = model_outputs["x"]  # [B, T+1, D]

                # Use CLS token
                cls_feature = x[:, 0]  # [B, D]

                # Normalize features for cosine similarity
                cls_feature = F.normalize(cls_feature, dim=1)

                print(f"Extracted feature shape: {cls_feature.shape}")
                print(f"Feature stats: Mean={cls_feature.mean().item():.4f}, Std={cls_feature.std().item():.4f}, Norm={cls_feature.norm().item():.4f}")

                # Note: Pretraining model cannot do classification directly
                print("\nNote: This is a pretraining model without classification head.")
                print("For classification, use a fine-tuned checkpoint from sft_4_AS20K or similar directory.")

            else:
                # Generic forward pass
                output = model(mel)
                if isinstance(output, dict):
                    if 'x' in output:
                        x = output['x'][:, 0]
                        print(f"Extracted feature shape: {x.shape}")
                    elif '_predictions' in output:
                        probs = output['_predictions']
                        values, indices = torch.topk(probs, args.top_k_prediction)
                        print("\n************ Acoustic Event Inference ************")
                        print("LABEL".ljust(30) + "PREDICTION")
                        for i, v in zip(indices[0], values[0]):
                            print(f"{vocab[i.item()]:<30} {v.item():.3f}")
                        print("**************************************************\n")
                        return
                else:
                    logits = output
                    probs = torch.sigmoid(logits)
                    values, indices = torch.topk(probs, args.top_k_prediction)
                    print("\n************ Acoustic Event Inference ************")
                    print("LABEL".ljust(30) + "PREDICTION")
                    for i, v in zip(indices[0], values[0]):
                        print(f"{vocab[i.item()]:<30} {v.item():.3f}")
                    print("**************************************************\n")

        except Exception as e:
            import traceback
            print(f"Inference failed: {e}")
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
