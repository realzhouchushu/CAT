#!/usr/bin/env python3
"""
检查评估数据和结果文件的匹配情况
"""
import json
import sys

def check_data_match(eval_jsonl, pred_file, gt_file):
    """检查评估数据与结果文件的匹配"""
    
    # 读取evaluation.jsonl获取所有audio_ids和references
    print("="*60)
    print("Checking evaluation data structure")
    print("="*60)
    
    audio_refs = {}  # {base_key: [ref1, ref2, ...]}
    with open(eval_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            key = data.get('key', '')
            base_key = key.rsplit('_', 1)[0] if '_' in key and key.split('_')[-1].isdigit() else key
            caption = data.get('target', data.get('caption', ''))
            
            if base_key not in audio_refs:
                audio_refs[base_key] = []
            audio_refs[base_key].append(caption)
    
    print(f"Total audios in evaluation.jsonl: {len(audio_refs)}")
    ref_counts = [len(refs) for refs in audio_refs.values()]
    print(f"References per audio: min={min(ref_counts)}, max={max(ref_counts)}, avg={sum(ref_counts)/len(ref_counts):.2f}")
    
    # 读取pred和gt文件
    pred_dict = {}
    gt_dict = {}
    
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                pred_dict[parts[0]] = parts[1]
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                gt_dict[parts[0]] = parts[1]
    
    print(f"\nPred file: {len(pred_dict)} entries")
    print(f"GT file: {len(gt_dict)} entries")
    
    # 检查匹配
    pred_keys = set(pred_dict.keys())
    gt_keys = set(gt_dict.keys())
    eval_keys = set(audio_refs.keys())
    
    print(f"\nKey matching:")
    print(f"  Pred keys matching eval keys: {len(pred_keys & eval_keys)}/{len(pred_keys)}")
    print(f"  GT keys matching eval keys: {len(gt_keys & eval_keys)}/{len(gt_keys)}")
    print(f"  Pred keys matching GT keys: {len(pred_keys & gt_keys)}/{len(pred_keys)}")
    
    if len(pred_keys & eval_keys) < len(pred_keys):
        missing = pred_keys - eval_keys
        print(f"\nWarning: {len(missing)} pred keys not in evaluation.jsonl")
        print(f"  First few: {list(missing)[:5]}")
    
    # 检查是否应该使用多参考评估
    print(f"\n" + "="*60)
    print("Recommendation:")
    print("="*60)
    
    if len(eval_keys) == len(pred_keys) and min(ref_counts) > 1:
        print("⚠️  evaluation.jsonl contains MULTIPLE references per audio")
        print("   Using multiple references may improve METEOR and other metrics")
        print("   Current script uses single reference from evaluation_single.jsonl")
        print("\n   To use multiple references, you need to:")
        print("   1. Match pred keys to base audio keys")
        print("   2. Group references by base audio key")
        print("   3. Use mult_references with all references for each audio")
    
    return audio_refs, pred_dict, gt_dict

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python check_evaluation_data.py <evaluation.jsonl> <pred_file> <gt_file>")
        sys.exit(1)
    
    eval_jsonl = sys.argv[1]
    pred_file = sys.argv[2]
    gt_file = sys.argv[3]
    
    check_data_match(eval_jsonl, pred_file, gt_file)

