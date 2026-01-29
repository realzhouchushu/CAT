#!/usr/bin/env python3
"""
评估脚本 - 支持多参考评估（使用evaluation.jsonl，每个audio有5个参考）
这应该与官方评估方式一致
"""
from aac_metrics import Evaluate
import json
import sys

def compute_wer_with_multiref(eval_jsonl, hyp_file):
    """
    使用多参考评估（evaluation.jsonl格式）
    """
    # 读取evaluation.jsonl获取所有audio的多参考
    audio_refs = {}  # {base_key: [ref1, ref2, ref3, ref4, ref5]}
    
    print("Loading references from evaluation.jsonl...")
    with open(eval_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            key = data.get('key', '')
            caption = data.get('target', data.get('caption', ''))
            
            # 处理key格式：有些是 "Santa Motor_1", "Santa Motor_2" 等
            # 提取base key（去掉_1, _2等后缀）
            if '_' in key and key.split('_')[-1].isdigit():
                base_key = key.rsplit('_', 1)[0]
            else:
                base_key = key
            
            if base_key not in audio_refs:
                audio_refs[base_key] = []
            audio_refs[base_key].append(caption)
    
    print(f"Loaded {len(audio_refs)} audios with multiple references")
    ref_counts = [len(refs) for refs in audio_refs.values()]
    print(f"References per audio: min={min(ref_counts)}, max={max(ref_counts)}, avg={sum(ref_counts)/len(ref_counts):.2f}")
    
    # 读取pred文件
    pred_dict = {}
    with open(hyp_file, 'r') as hyp_reader:
        for line in hyp_reader:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                pred_dict[key] = value
    
    print(f"Loaded {len(pred_dict)} predictions")
    
    # 匹配pred和references（使用pred的key顺序）
    candidates = []
    mult_references = []
    missing_keys = []
    
    for key in pred_dict.keys():
        if key in audio_refs:
            candidates.append(pred_dict[key])
            mult_references.append(audio_refs[key])  # 使用所有参考
        else:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys in pred but not in evaluation.jsonl. First few: {missing_keys[:5]}")
    
    if len(candidates) == 0:
        print("Error: No matching keys found!")
        return
    
    print(f'Using {len(candidates)} matched predictions with multiple references')
    
    # 评估
    evaluator = Evaluate(metrics=["meteor", "cider_d", "spice", "spider", "spider_fl", "fense"])
    
    print("Computing metrics with multiple references...")
    corpus_scores, sent_scores = evaluator(candidates, mult_references)
    
    # 格式化输出
    print("\n" + "="*70)
    print("Evaluation Results (Clotho evaluation split, MULTI-REFERENCE):")
    print("="*70)
    main_metrics = {
        "meteor": "MT (METEOR)",
        "cider_d": "CD (CIDEr)", 
        "spice": "SC (SPICE)",
        "spider": "SD (SPIDEr)",
        "spider_fl": "SF (SPIDEr-FL)",
        "fense": "FS (FENSE)"
    }
    
    results = {}
    for metric_key, metric_name in main_metrics.items():
        if metric_key in corpus_scores:
            value = corpus_scores[metric_key]
            score = value.item() if hasattr(value, 'item') else float(value)
            score_pct = score * 100
            results[metric_key] = score_pct
            print(f"{metric_name:20s}: {score_pct:6.2f}%")
    
    
    print("\nFull corpus scores (raw):")
    print(corpus_scores)

if __name__ == '__main__':
    """
    example:
    python ~/codes/2506/EAT/SLAM-LLM/src/slam_llm/utils/compute_aac_metrics_multi_ref.py ~/data/aac-datasets-raw-meta/clotho/evaluation.jsonl ~/exp/aac/eat_offical_pretrain/aac_epoch_1_step_4500/decode_beam2-8_pred
    """
    if len(sys.argv) != 3:
        print("usage: python compute_aac_metrics_multi_ref.py <evaluation.jsonl> <hyp_file>")
        print("  Note: Uses multiple references from evaluation.jsonl (5 refs per audio)")
        sys.exit(0)
    
    eval_jsonl = sys.argv[1]
    hyp_file = sys.argv[2]
    compute_wer_with_multiref(eval_jsonl, hyp_file)

