from aac_metrics import Evaluate
import json
import sys

def compute_wer(ref_file,
                hyp_file,
                use_multiref=False,
                eval_jsonl=None):
    """
    Args:
        ref_file: GT文件路径（tab分隔：key\tcaption）
        hyp_file: Pred文件路径（tab分隔：key\tcaption）
        use_multiref: 是否使用多参考评估（从evaluation.jsonl加载所有参考）
        eval_jsonl: evaluation.jsonl路径（当use_multiref=True时必需）
    """
    
    # 读取pred文件
    pred_dict = {}
    with open(hyp_file, 'r') as hyp_reader:
        for line in hyp_reader:
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                pred_dict[key] = value
    
    if use_multiref and eval_jsonl:
        # 多参考评估：从evaluation.jsonl加载所有参考
        print("Using MULTI-REFERENCE evaluation (from evaluation.jsonl)")
        audio_refs = {}
        
        with open(eval_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                key = data.get('key', '')
                caption = data.get('target', data.get('caption', ''))
                
                # 处理key格式：去掉_1, _2等后缀得到base key
                if '_' in key and key.split('_')[-1].isdigit():
                    base_key = key.rsplit('_', 1)[0]
                else:
                    base_key = key
                
                if base_key not in audio_refs:
                    audio_refs[base_key] = []
                audio_refs[base_key].append(caption)
        
        # 匹配pred和references
        candidates = []
        mult_references = []
        
        for key in pred_dict.keys():
            if key in audio_refs:
                candidates.append(pred_dict[key])
                mult_references.append(audio_refs[key])  # 使用所有5个参考
            else:
                print(f"Warning: key '{key}' not found in evaluation.jsonl")
        
        print(f"Loaded {len(candidates)} predictions with {len(mult_references[0]) if mult_references else 0} references per audio")
    else:
        # 单参考评估：从ref_file读取
        print("Using SINGLE-REFERENCE evaluation (from ref_file)")
        gt_dict = {}
        with open(ref_file, 'r') as ref_reader:
            for line in ref_reader:
                parts = line.strip().split('\t', 1)
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    gt_dict[key] = value
        
        # 匹配pred和gt
        candidates = []
        mult_references = []
        
        for key in pred_dict.keys():
            if key in gt_dict:
                candidates.append(pred_dict[key])
                mult_references.append([gt_dict[key]])  # 单参考
            else:
                print(f"Warning: key '{key}' not found in ref_file")
    
    if len(candidates) == 0:
        print("Error: No matching predictions found!")
        return
    
    print('Used lines:', len(candidates))
    
    evaluator = Evaluate(metrics=["meteor", "cider_d", "spice", "spider", "spider_fl", "fense"])
    """
        "bert_score": BERTScoreMRefs,
        "bleu": BLEU,
        "bleu_1": BLEU1,
        "bleu_2": BLEU2,
        "bleu_3": BLEU3,
        "bleu_4": BLEU4,
        "clap_sim": CLAPSim,
        "cider_d": CIDErD,
        "fer": FER,
        "fense": FENSE,
        "mace": MACE,
        "meteor": METEOR,
        "rouge_l": ROUGEL,
        "sbert_sim": SBERTSim,
        "spice": SPICE,
        "spider": SPIDEr,
        "spider_max": SPIDErMax,
        "spider_fl": SPIDErFL,
        "vocab": Vocab,
    }"""

    print("Computing metrics...")
    corpus_scores, sent_scores = evaluator(candidates, mult_references)
    
    # 格式化输出
    print("\n" + "="*70)
    print("Evaluation Results (Clotho evaluation split):")
    print("="*70)
    main_metrics = {
        "meteor": "MT (METEOR)",
        "cider_d": "CD (CIDEr)", 
        "spice": "SC (SPICE)",
        "spider": "SD (SPIDEr)",
        "spider_fl": "SF (SPIDEr-FL)",
        "fense": "FS (FENSE)"
    }
    
    for metric_key, metric_name in main_metrics.items():
        if metric_key in corpus_scores:
            value = corpus_scores[metric_key]
            score = value.item() if hasattr(value, 'item') else float(value)
            score_pct = score * 100
            print(f"{metric_name:20s}: {score_pct:6.2f}%")
    
    print("="*70)
    print("\nOfficial SLAM-AAC results (for comparison):")
    print("  MT: 19.7%  |  CD: 51.5%  |  SC: 14.8%  |  SD: 33.2%  |  SF: 33.0%  |  FS: 54.0%")
    print("="*70)
    
    # 也输出原始格式
    print("\nFull corpus scores (raw):")
    print(corpus_scores)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute AAC metrics')
    parser.add_argument('ref_file', help='Reference file (GT)')
    parser.add_argument('hyp_file', help='Hypothesis file (Predictions)')
    parser.add_argument('--multiref', action='store_true', 
                       help='Use multi-reference evaluation (requires --eval-jsonl)')
    parser.add_argument('--eval-jsonl', type=str, default=None,
                       help='Path to evaluation.jsonl for multi-reference evaluation')
    args = parser.parse_args()
    
    if args.multiref and not args.eval_jsonl:
        print("Error: --multiref requires --eval-jsonl")
        print("\nUsage examples:")
        print("  Single reference: python compute_aac_metrics.py gt.txt pred.txt")
        print("  Multi reference:  python compute_aac_metrics.py gt.txt pred.txt --multiref --eval-jsonl evaluation.jsonl")
        sys.exit(1)
    
    compute_wer(args.ref_file, args.hyp_file, 
                use_multiref=args.multiref, 
                eval_jsonl=args.eval_jsonl)