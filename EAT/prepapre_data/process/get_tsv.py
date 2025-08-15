#!/usr/bin/env python3
"""
AudioSet unbalanced training data manifest generation script.
"""

import argparse
import glob
import os
import random
import soundfile
from tqdm import tqdm
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="Generate TSV manifest for AudioSet unbalanced training data")
    parser.add_argument(
        "root", 
        metavar="DIR", 
        help="root directory containing audio files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1), default 0"
    )
    parser.add_argument(
        "--dest", 
        default=".", 
        type=str, 
        metavar="DIR", 
        help="output directory for manifest files"
    )
    parser.add_argument(
        "--ext", 
        default="wav", 
        type=str, 
        metavar="EXT", 
        help="audio file extension to look for (default: wav)"
    )
    parser.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        metavar="N", 
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "--path-must-contain",
        default=None, 
        type=str, 
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included"
    )
    parser.add_argument(
        "--max-files",
        default=None,
        type=int,
        metavar="N",
        help="maximum number of files to process (for testing)"
    )
    return parser


def get_audio_info(file_path):
    """获取音频文件信息"""
    try:
        info = soundfile.info(file_path)
        return info.frames, info.samplerate, info.duration
    except Exception as e:
        print(f"Warning: Could not read audio file {file_path}: {e}")
        return None, None, None


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    rand = random.Random(args.seed)

    # 创建输出目录
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # 获取根目录的绝对路径
    root_path = os.path.realpath(args.root)
    print(f"Processing directory: {root_path}")
    
    audio_files = []
    for fname in tqdm(os.listdir(root_path), desc="Processing files"):
        file_path = os.path.join(root_path, fname)
        # # 已经确保文件夹下全是有用音频直接
        # if args.ext not in file:
        #     continue
        # if os.path.isdir(file_path):
        #     continue
        
        # # 检查路径过滤条件
        # if args.path_must_contain and args.path_must_contain not in file_path:
        #     continue
            
        # 获取音频信息
        frames, sr, duration = get_audio_info(file_path)
        if frames is not None:
            audio_files.append((fname, frames))
    
    print(f"Found {len(audio_files)} valid audio files")
    
    # 限制文件数量（用于测试）
    if args.max_files and len(audio_files) > args.max_files:
        audio_files = audio_files[:args.max_files]
        print(f"Limited to {len(audio_files)} files for testing")

    # # 随机打乱文件顺序
    # rand.shuffle(audio_files)
    
    # 计算验证集大小
    valid_size = int(len(audio_files) * args.valid_percent)
    train_size = len(audio_files) - valid_size
    
    print(f"Training set: {train_size} files")
    print(f"Validation set: {valid_size} files")
    
    # 写入训练集manifest
    train_manifest_path = os.path.join(args.dest, "train.tsv")
    with open(train_manifest_path, "w", encoding="utf-8") as train_f:
        # 第一行：公共路径
        print(root_path, file=train_f)
        
        # 训练集文件
        for fname, frames in tqdm(audio_files[:train_size], desc="Writing train manifest"):
            print(f"{fname}\t{frames}", file=train_f)
    
    print(f"Training manifest saved to: {train_manifest_path}")
    
    # 写入验证集manifest（如果指定了验证集）
    if args.valid_percent > 0:
        valid_manifest_path = os.path.join(args.dest, "valid.tsv")
        with open(valid_manifest_path, "w", encoding="utf-8") as valid_f:
            # 第一行：公共路径
            print(root_path, file=valid_f)
            
            # 验证集文件
            for fname, frames in tqdm(audio_files[train_size:], desc="Writing valid manifest"):
                print(f"{fname}\t{frames}", file=valid_f)
        
        print(f"Validation manifest saved to: {valid_manifest_path}")
    
    # 生成统计信息
    stats_path = os.path.join(args.dest, "manifest_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as stats_f:
        stats_f.write(f"AudioSet Unbalanced Training Data Manifest Statistics\n")
        stats_f.write(f"=" * 50 + "\n")
        stats_f.write(f"Root directory: {root_path}\n")
        stats_f.write(f"Total files found: {len(audio_files)}\n")
        stats_f.write(f"Training files: {train_size}\n")
        stats_f.write(f"Validation files: {valid_size}\n")
    
    print(f"Statistics saved to: {stats_path}")
    print("Manifest generation completed successfully!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)