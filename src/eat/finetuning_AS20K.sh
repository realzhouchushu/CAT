#!/bin/bash
model_model_path=/opt/gpfs/home/chushu/exp/eat/pre_4_AS2M/disp_0_2025-09-24_14-17-47/checkpoint_last.pt

model_linear_layer=${1}
echo "model_linear_layer: ${model_linear_layer}"
SAVE_DIR_ROOT=/opt/gpfs/home/chushu/exp/eat/sft_4_AS20k_w_disp_CLS_lw1_${model_linear_layer}
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
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir=${checkpoint_save_dir} \
    checkpoint.restore_file=${checkpoint_restore_file} \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=48 \
    dataset.num_workers=24 \
    dataset.data_buffer_size=48 \
    task.data=/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS20k \
    task.target_length=1024 \
    task.roll_aug=true \
    task.load_clap_emb=false \
    +task.load_source_file=true \
    +task.load_mel_file=false \
    +model.linear_classifier=false \
    +model.linear_layer=${model_linear_layer} \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=${model_model_path} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN # CLS_TOKEN num_workers: 6 data_buffer_size: 10