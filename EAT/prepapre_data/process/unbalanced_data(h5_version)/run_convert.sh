#!/bin/bash

# CSV到TSV转换脚本运行器
# 使用方法: ./run_convert.sh [csv_file] [output_dir]

set -e  # 遇到错误时退出

# 默认参数
CSV_FILE="/opt/gpfs/data/raw_data/AudioSet/unbalanced_train/waveform/waveform.csv"
OUTPUT_DIR="."
SAMPLE_RATE=32000
VALID_PERCENT=0
SEED=42

# 检查命令行参数
if [ $# -ge 1 ]; then
    CSV_FILE="$1"
fi

if [ $# -ge 2 ]; then
    OUTPUT_DIR="$2"
fi

# 检查CSV文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "错误: CSV文件不存在: $CSV_FILE"
    echo "请检查文件路径是否正确"
    exit 1
fi

# 显示参数
echo "=== CSV到TSV转换脚本 ==="
echo "CSV文件: $CSV_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "采样率: $SAMPLE_RATE Hz"
echo "验证集比例: $VALID_PERCENT"
echo "随机种子: $SEED"
echo "=========================="

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import pandas, h5py, numpy, tqdm" 2>/dev/null || {
    echo "错误: 缺少必要的Python包"
    echo "请安装: pip install pandas h5py numpy tqdm"
    exit 1
}

# 运行转换
echo "开始转换..."
python3 convert_csv_to_tsv.py \
    "$CSV_FILE" \
    "$OUTPUT_DIR" \
    --sample-rate "$SAMPLE_RATE" \
    --valid-percent "$VALID_PERCENT" \
    --seed "$SEED"

echo "转换完成！"
echo "输出文件:"
ls -la "$OUTPUT_DIR"

# 显示统计信息
if [ -f "$OUTPUT_DIR/audio_stats.txt" ]; then
    echo ""
    echo "=== 音频统计信息 ==="
    cat "$OUTPUT_DIR/audio_stats.txt"
fi
