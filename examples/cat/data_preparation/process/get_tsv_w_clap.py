#!/usr/bin/env python3
"""
Script to process TSV files and CLAP embeddings to generate JSON output.
Matches wav files with their corresponding CLAP embedding files based on basename.
"""

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import shutil


def parse_tsv_file(tsv_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Parse TSV file to extract common path and wav file information.
    
    Args:
        tsv_path: Path to the TSV file
        
    Returns:
        Tuple of (common_path, list of (relative_path, frame_size))
    """
    wav_files = []
    common_path = ""
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            if line_num == 0:
                # First line contains common path
                common_path = line
            else:
                # Subsequent lines contain relative_path and frame_size
                parts = line.split('\t')
                if len(parts) >= 2:
                    relative_path = parts[0]
                    try:
                        frame_size = int(parts[1])
                        wav_files.append((relative_path, frame_size))
                    except ValueError:
                        print(f"Warning: Invalid frame size on line {line_num + 1}: {parts[1]}")
    
    return common_path, wav_files


def get_clap_files(clap_folder: str) -> Dict[str, str]:
    """
    Get all .npy files from CLAP folder and create a mapping from basename to full path.
    
    Args:
        clap_folder: Path to the folder containing CLAP embeddings
        
    Returns:
        Dictionary mapping basename to full CLAP file path
    """
    clap_files = {}
    
    if not os.path.exists(clap_folder):
        print(f"Warning: CLAP folder does not exist: {clap_folder}")
        return clap_files
    
    for root, dirs, files in tqdm(os.walk(clap_folder), desc="Scanning CLAP embeddings folder"):
        for file in tqdm(files, desc="Processing CLAP files"):
            if file.endswith('.npy'):
                basename = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                clap_files[basename] = full_path
    
    return clap_files

def get_mel_files(mel_folder: str) -> Dict[str, str]:
    """
    Get all .npy files from mel folder and create a mapping from basename to full path.
    """
    mel_files = {}

    if not os.path.exists(mel_folder):
        print(f"Warning: mel folder does not exist: {mel_folder}")
        return mel_files
    
    for root, dirs, files in tqdm(os.walk(mel_folder), desc="Scanning mel folder"):
        for file in tqdm(files, desc="Processing mel files"):
            if file.endswith('.npy'):
                basename = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                mel_files[basename] = full_path
    return mel_files

def match_files(wav_files: List[Tuple[str, int]], clap_files: Dict[str, str], mel_files: Dict[str, str],
                common_path: str) -> Tuple[List[Dict], int, int, int]:
    """
    Match wav files with CLAP files and count statistics.
    
    Args:
        wav_files: List of (relative_path, frame_size) tuples
        clap_files: Dictionary mapping basename to CLAP file path
        common_path: Common path for wav files
        
    Returns:
        Tuple of (matched_files, only_wav_count, only_clap_count, both_count)
    """
    # print(f"mel_files: {mel_files}")
    print(f"len(mel_files): {len(mel_files)}")
    matched_files = []
    only_wav_count = 0
    only_clap_count = 0
    both_count = 0
    
    # Track which CLAP files have been matched
    matched_clap_basenames = set()
    
    for relative_path, frame_size in wav_files:
        # Get basename without extension
        wav_basename = os.path.splitext(os.path.basename(relative_path))[0]
        
        if wav_basename in clap_files and wav_basename in mel_files:
            # Both wav and CLAP exist
            dict = {
                "wav_path": os.path.join(common_path, relative_path),
                "clap_path": clap_files[wav_basename],
                "mel_path": mel_files[wav_basename],
                "num_frame": frame_size
            }
            matched_files.append(dict)
            matched_clap_basenames.add(wav_basename)
            both_count += 1
        else:
            # Only wav exists
            only_wav_count += 1
    
    # Count CLAP files that don't have matching wav files
    for basename in clap_files:
        if basename not in matched_clap_basenames:
            only_clap_count += 1
    
    return matched_files, only_wav_count, only_clap_count, both_count


def save_json_output(output_path: str, matched_files: List[Dict], 
                    common_path: str, clap_folder: str):
    """
    Save the matched files to JSON with metadata comments.
    
    Args:
        output_path: Path to save the JSON file
        matched_files: List of matched file dictionaries
        common_path: Common path for wav files
        clap_folder: Path to CLAP embeddings folder
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write metadata as comments
        # f.write(f"# audio_path: {common_path}\n")
        # f.write(f"# embedding_path: {clap_folder}\n")
        
        # Write JSON data
        json.dump(matched_files, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Process TSV files and CLAP embeddings')
    parser.add_argument('--tsv_path', 
                       default='~/data/audioset/16k_wav_tsv/unbal_train.tsv',
                       help='Path to input TSV file')
    parser.add_argument('--clap_folder', 
                       default='~/raw_datas/features/audioset/clap_embs',
                       help='Path to CLAP embeddings folder')
    # options: ['/opt/gpfs/home/chushu/data/features/ast_features/mlp_head_in','/opt/gpfs/home/chushu/data/features/ast_features/mlp_head_out', '/opt/gpfs/home/chushu/data/features/clap_features/clap_embs/unbalanced_train_segments']
    parser.add_argument('--output_path', 
                       default='~/data/audioset/meta_w_feature/unbal_train_clap.json',
                       help='Path to output JSON file')
    
    args = parser.parse_args()
    
    print(f"Processing TSV file: {args.tsv_path}")
    print(f"CLAP embeddings folder: {args.clap_folder}")
    print(f"Output JSON file: {args.output_path}")
    print("-" * 50)
    
    # Parse TSV file
    print("Parsing TSV file...")
    common_path, wav_files = parse_tsv_file(args.tsv_path)
    print(f"Found {len(wav_files)} wav files")
    print(f"Common path: {common_path}")
    
    # Get CLAP files
    print("Scanning CLAP embeddings folder...")
    clap_files = get_clap_files(args.clap_folder)
    print(f"Found {len(clap_files)} CLAP embedding files")

    # Get mel files
    print("Scanning mel folder...")
    mel_files = get_mel_files(args.mel_folder)
    print(f"Found {len(mel_files)} mel files")

    # Match files
    print("Matching wav files with CLAP embeddings...")
    matched_files, only_wav_count, only_clap_count, both_count = match_files(
        wav_files, clap_files, mel_files, common_path
    )
    
    # Print statistics
    print("\n" + "=" * 50)
    print("MATCHING STATISTICS:")
    print(f"Files with both wav and CLAP: {both_count}")
    print(f"Files with only wav: {only_wav_count}")
    print(f"Files with only CLAP: {only_clap_count}")
    print(f"Total wav files: {len(wav_files)}")
    print(f"Total CLAP files: {len(clap_files)}")
    print("=" * 50)
    
    # Save output
    print(f"\nSaving {len(matched_files)} matched files to JSON...")
    save_json_output(args.output_path, matched_files, common_path, args.clap_folder)
    print(f"Successfully saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TSV files and CLAP embeddings')
    parser.add_argument('--tsv_path', 
                       default='/opt/gpfs/home/chushu/data/audioset/16k_wav_tsv/bal_train.tsv',
                       help='Path to input TSV file')
    parser.add_argument('--clap_folder', 
                       default='/opt/gpfs/home/chushu/data/features/eat_clap_feature/1',
                       help='Path to CLAP embeddings folder') # !!!Modify
    parser.add_argument('--mel_folder', 
                       default='/opt/gpfs/home/chushu/data/features/eat_clap_feature/train_mel',
                       help='Path to mel folder')
    # options: ['/opt/gpfs/home/chushu/data/features/ast_features/mlp_head_in','/opt/gpfs/home/chushu/data/features/ast_features/mlp_head_out', '/opt/gpfs/home/chushu/data/features/clap_features/clap_embs/unbalanced_train_segments']
    parser.add_argument('--output_path', 
                       default='/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/1/train.json',
                       help='Path to output JSON file') # !!!Modify
    
    args = parser.parse_args()
    for i in range(0, 14):
        if i == 13:
            i = 'eval'
            args.tsv_path = '/opt/gpfs/home/chushu/data/audioset/16k_wav_tsv/eval.tsv'
            args.mel_folder = '/opt/gpfs/home/chushu/data/features/eat_clap_feature/eval_mel'
        print("-" * 50)
        print(f"Processing layer {i}...")
        args.clap_folder = f'/opt/gpfs/home/chushu/data/features/eat_clap_feature/{i}' # !!!Modify
        args.output_path = f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}/train.json' # !!!Modify
        os.makedirs(f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}', exist_ok=True) # !!!Modify
        shutil.copy('/opt/gpfs/home/chushu/data/audioset/16k_wav_tsv/bal_train.lbl', f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}/train.lbl') # !!!Modify
        shutil.copy('/opt/gpfs/home/chushu/data/audioset/16k_wav_tsv/eval.lbl', f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}/eval.lbl') # !!!Modify
        shutil.copy('/opt/gpfs/home/chushu/data/audioset/label_descriptors.csv', f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}/label_descriptors.csv') # !!!Modify
        main(args)
        if i == 'eval':
            for j in range(0, 13):
                shutil.copy(f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{i}/train.json', f'/opt/gpfs/home/chushu/data/audioset/setting/LINEAR_AS20k_EAT_CLAP/{j}/eval.json') # !!!Modify
