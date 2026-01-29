#!/bin/bash
model_model_path=~/hubs/models/huggingface/zhouchushu/tmp_model_store/pre_4_AS2M/default_0_2025-09-20_15-33-21/checkpoint_last.pt

model_linear_layer=${1:-0}
model_add_bottleneck=${2:-false}
echo "model_linear_layer: ${model_linear_layer}"
SAVE_DIR_ROOT=~/exp/cat/sft_4_AS2M/default_0_41_400000

# 从 model_model_path 提取父目录名与文件名
parent_dir="$(basename -- "$(dirname -- "$model_model_path")")"
ckpt_name="$(basename -- "$model_model_path")"
# 组合保存目录并确保存在
checkpoint_save_dir="${SAVE_DIR_ROOT%/}/${parent_dir}"
mkdir -p -- "$checkpoint_save_dir"
checkpoint_restore_file="${checkpoint_save_dir%/}/${ckpt_name}"
echo "checkpoint_save_dir: ${checkpoint_save_dir}"
echo "checkpoint_restore_file: ${checkpoint_restore_file}"

device=0

CUDA_VISIBLE_DEVICES=${device} python fairseq_cli/hydra_train.py -m \
    --config-dir examples/cat/config \
    --config-name finetuning  \
    common.user_dir=examples/cat \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=96 \
    dataset.num_workers=24 \
    dataset.data_buffer_size=96 \
    task.data=~/codes/2506/CAT/examples/cat/data_manifest/SFT_AS2M \
    task.h5_format=false \
    task.AS2M_finetune=true \
    task.weights_file=~/codes/2506/CAT/examples/cat/data_manifest/SFT_AS2M/weights.csv \
    task.load_clap_emb=false \
    task.target_length=1024 \
    task.roll_aug=true \
    model.model_path=${model_model_path} \
    +model.add_bottleneck=${model_add_bottleneck} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \