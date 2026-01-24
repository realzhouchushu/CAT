import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

# --- 配置 ---

# 1. 本地数据集根目录 (用于构建音频索引)
DATASET_ROOTS: Dict[str, Path] = {
    "AudioCaps": Path("/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/audioset/wav_16k"),
    "Clotho": Path("/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/aac-datasets/CLOTHO_v2.1"),
    "MACS": Path("/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/aac-datasets/MACS"),
    "WavCaps": Path("/inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/aac-datasets/WavCaps"),
}

# 2. JSONL 文件搜索和输出配置
# 要搜索所有 JSONL 文件的根目录
INPUT_BASE_DIR = Path("/inspire/hdd/global_user/zhouchushu-253108120180/data/aac-datasets-raw-meta")
# 预期的输出路径，替换 INPUT_BASE_DIR 的顶层目录名 'aac-datasets'
OUTPUT_BASE_DIR = Path("/inspire/hdd/global_user/zhouchushu-253108120180/data/acc-datasets")

# 常见音频文件扩展名，用于递归搜索
AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.m4a', '.ogg')

# WavCaps 的子集名称，用于构造唯一键
WAVCAPS_SUBSETS = ["AudioSet_SL", "BBC_Sound_Effects", "FreeSound", "SoundBible"]

# Clotho 的子集名称 (包含 development, evaluation, validation, test)
CLOTHO_SUBSETS = ["development", "evaluation", "validation", "test"]

# 全局文件名到绝对路径的映射 (Index Map)
FILE_PATH_INDEX: Dict[str, Path] = {}


def build_audio_file_index() -> Dict[str, Path]:
    """
    递归遍历所有数据集根目录，建立文件名到绝对路径的映射。
    对于 WavCaps 和 Clotho，键为 '子集名称/文件名'。
    """
    print("--- 步骤 1/3: 正在构建音频文件索引... ---")
    
    total_files_indexed = 0
    for name, root_path in DATASET_ROOTS.items():
        if not root_path.is_dir():
            print(f"[警告] 数据集根目录不存在或不是目录: {name} ({root_path})，跳过。")
            continue
            
        count = 0
        # 使用 os.walk 递归遍历所有子目录
        for dirpath, _, filenames in os.walk(root_path):
            current_path = Path(dirpath)
            for filename in filenames:
                if filename.lower().endswith(AUDIO_EXTENSIONS):
                    full_path = current_path / filename
                    
                    file_key = filename # 默认键：文件名
                    
                    # --- WavCaps 特殊处理：使用子集名称作为前缀 ---
                    if name == "WavCaps":
                        try:
                            # 查找 WavCaps/Audio/ 下的子集目录名
                            relative_to_root = full_path.relative_to(root_path) 
                            
                            subset_name = ""
                            # 遍历路径组件，找到匹配的子集名
                            for part in relative_to_root.parts:
                                if part in WAVCAPS_SUBSETS:
                                    subset_name = part
                                    break
                            
                            if subset_name:
                                file_key = f"{subset_name}/{filename}"
                            else:
                                # 如果在 WavCaps 目录下但找不到子集，仍使用文件名
                                file_key = filename
                        except Exception:
                            print(f"[警告] WavCaps 文件路径结构异常，使用文件名作为键: {full_path}")
                            file_key = filename
                    
                    # --- Clotho 特殊处理：使用子集名称作为前缀 ---
                    elif name == "Clotho":
                        try:
                            # 查找 Clotho/clotho_audio_files/ 下的子集目录名 (development/evaluation/validation)
                            relative_to_root = full_path.relative_to(root_path)
                            
                            subset_name = ""
                            # 遍历路径组件，找到匹配的子集名
                            for part in relative_to_root.parts:
                                if part in CLOTHO_SUBSETS:
                                    subset_name = part
                                    break
                            
                            if subset_name:
                                file_key = f"{subset_name}/{filename}"
                            else:
                                # 如果在 Clotho 目录下但找不到子集，仍使用文件名
                                file_key = filename
                        except Exception:
                            print(f"[警告] Clotho 文件路径结构异常，使用文件名作为键: {full_path}")
                            file_key = filename
                    
                    # --- 结束特殊处理 ---

                    
                    if file_key in FILE_PATH_INDEX:
                        # 冲突警告：如果键冲突，将使用后找到的路径
                        print(f"[警告] 文件名冲突: {file_key}。上一个路径: {FILE_PATH_INDEX[file_key]}，当前路径: {full_path}。将使用后找到的路径。")

                    FILE_PATH_INDEX[file_key] = full_path
                    count += 1
        
        total_files_indexed += count
        print(f"✅ {name}: 索引完成，找到 {count} 个音频文件。")

    print(f"索引构建完成。总共找到 {total_files_indexed} 个唯一音频文件。")
    return FILE_PATH_INDEX

def find_jsonl_files(base_dir: Path) -> List[Path]:
    """
    递归查找指定目录下所有的 .jsonl 文件。
    """
    print(f"\n--- 步骤 2/3: 正在搜索 {base_dir} 下的所有 .jsonl 文件... ---")
    # 使用 rglob 进行递归搜索
    jsonl_files = list(base_dir.rglob("*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个 .jsonl 文件进行处理。")
    return jsonl_files


def process_single_jsonl(input_path: Path, index_map: Dict[str, Path]):
    """
    读取单个输入 JSONL 文件，更新 'source' 字段，并写入输出文件。
    """
    print(f"\n--- 正在处理文件: {input_path} ---")

    # 1. 确定输出文件路径
    # 获取输入路径相对于 INPUT_BASE_DIR 的部分
    try:
        relative_path = input_path.relative_to(INPUT_BASE_DIR)
    except ValueError:
        print(f"[错误] 输入文件 {input_path} 不在预期的搜索根目录 {INPUT_BASE_DIR} 下。跳过此文件。")
        return

    # 拼接新的输出路径
    OUTPUT_JSONL_PATH = OUTPUT_BASE_DIR / relative_path
    
    # 2. 确保输出目录存在
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"将写入文件到: {OUTPUT_JSONL_PATH}")

    # 3. 处理文件
    total_lines = 0
    updated_lines = 0
    not_found_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[警告] 行 {total_lines} 无法解析: {line.strip()[:80]}...")
                continue
            
            # 从原始 source 路径中提取文件名
            original_source = data.get("source", "")
            if not original_source:
                # 保持原样写入，并跳过处理
                outfile.write(line)
                continue

            original_path = Path(original_source)
            filename = original_path.name
            
            file_key = filename # 默认键：文件名
            
            # --- 查找键特殊处理：根据 original_source 字符串判断数据集类型 ---
            
            # 1. WavCaps 查找键特殊处理
            if "WavCaps" in original_source: 
                subset_name = ""
                # 在原始路径组件中查找匹配的子集名
                for part in original_path.parts:
                    if part in WAVCAPS_SUBSETS:
                        subset_name = part
                        break
                    # FIX: 处理 JSONL 路径中可能存在的 '_flac' 后缀
                    for ss in WAVCAPS_SUBSETS:
                        if part == f"{ss}_flac":
                            subset_name = ss # 提取纯净的子集名称
                            break
                    if subset_name:
                        break
                
                if subset_name:
                    file_key = f"{subset_name}/{filename}"

            # 2. Clotho 查找键特殊处理 (修复：检查字符串中是否包含 "Clotho")
            elif "Clotho" in original_source: 
                subset_name = ""
                # 在原始路径组件中查找匹配的子集名
                for part in original_path.parts:
                    if part in CLOTHO_SUBSETS:
                        subset_name = part
                        break

                if subset_name:
                    file_key = f"{subset_name}/{filename}"
            # --- 结束查找键特殊处理 ---

            
            # 在索引中查找新的绝对路径
            new_path = index_map.get(file_key)
            
            if new_path:
                # 更新 source 字段为本地绝对路径
                data["source"] = str(new_path)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                updated_lines += 1
            else:
                # 如果找不到，保持原样写入并记录
                not_found_count += 1
                outfile.write(line)
                
    print(f"    总行数: {total_lines}")
    print(f"    ✅ 成功匹配并更新的行数: {updated_lines}")
    print(f"    ❌ 找不到对应音频文件的行数 (保持原路径): {not_found_count}")


if __name__ == "__main__":
    if not INPUT_BASE_DIR.exists():
        print(f"[致命错误] 输入 JSONL 搜索目录不存在: {INPUT_BASE_DIR}")
        sys.exit(1)

    # 1. 构建音频文件索引
    audio_index = build_audio_file_index()

    # 2. 查找所有 JSONL 文件
    jsonl_files_to_process = find_jsonl_files(INPUT_BASE_DIR)

    # 3. 循环处理每个 JSONL 文件
    if jsonl_files_to_process:
        print("\n--- 步骤 3/3: 循环处理所有 JSONL 文件 ---")
        for file_path in jsonl_files_to_process:
            process_single_jsonl(file_path, audio_index)
    else:
        print("[警告] 未找到任何 .jsonl 文件，脚本终止。")
        
    print("\n脚本执行完毕。请检查输出文件以确认路径是否正确。")
