#!/bin/bash

# CAT Inference Script for Fairseq Framework
#
# 使用说明:
# 1. 预训练模型特征提取 (当前配置)
#    bash inference.sh
#
# 2. 微调模型分类推理
#    修改 CHECKPOINT_DIR 为微调过的 checkpoint 路径
#    例如：/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS20K/xxx/checkpoint_best.pt
#
# 环境激活
source /inspire/hdd/global_user/zhouchushu-253108120180/miniconda3/etc/profile.d/conda.sh && conda activate fairseq

# ===== 配置文件路径 =====
SOURCE_FILE="/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/CAT/examples/cat/inference/test.wav"
LABEL_FILE="/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/CAT/examples/cat/inference/labels.csv"
MODEL_DIR="/inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/CAT/examples/cat"

# ===== Checkpoint 路径 =====
# 微调模型 (用于分类推理) - 默认使用 SFT 4 AS2M 的 checkpoint
CHECKPOINT_DIR="/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/sft_4_AS2M/conv_422_lw1_layer11_llayer0/conv_clap_0_2025-09-23_14-49-45/checkpoint_best.pt"

# 预训练模型 (仅用于特征提取，取消注释以使用):
# CHECKPOINT_DIR="/inspire/hdd/global_user/zhouchushu-253108120180/exp/eat/pre_4_AS2M/conv_clap_0/checkpoint_41_400000.pt"

# ===== 推理参数 =====
TARGET_LENGTH=1024        # 目标 mel 谱长度 (时间维度)
TOP_K=5                   # 输出 Top-K 预测结果
NORM_MEAN=-4.268          # AudioSet 全局归一化均值
NORM_STD=4.569            # AudioSet 全局归一化标准差
FRAMEWORK="fairseq"       # 模型框架
PREDICTION_MODE="CLS_TOKEN"  # 预测模式：CLS_TOKEN, MEAN_POOLING, FEATURE

# ===== 运行推理 =====
python ${MODEL_DIR}/inference/inference.py \
    --source_file ${SOURCE_FILE} \
    --label_file ${LABEL_FILE} \
    --model_dir ${MODEL_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --target_length ${TARGET_LENGTH} \
    --top_k_prediction ${TOP_K} \
    --norm_mean ${NORM_MEAN} \
    --norm_std ${NORM_STD} \
    --framework ${FRAMEWORK} \
    --prediction_mode ${PREDICTION_MODE}
