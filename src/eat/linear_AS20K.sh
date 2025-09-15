#!/bin/bash
model_model_path=/opt/gpfs/home/chushu/exp/eat/pre_4_AS2M/clap_0_2025-08-27_09-23-59/checkpoint_last.pt

# model_linear_layer=${1:-1}
model_linear_layer=7
optimization_lr='[0.005]'
optimizer_groups_default_lr_float=0.005

SAVE_DIR_ROOT=/opt/gpfs/home/chushu/exp/eat/linear_4_AS20k_w_clap_CLS
# 从 model_model_path 提取父目录名与文件名
parent_dir="$(basename -- "$(dirname -- "$model_model_path")")"
ckpt_name="$(basename -- "$model_model_path")"
# 组合保存目录并确保存在
checkpoint_save_dir="${SAVE_DIR_ROOT%/}/${parent_dir}/${model_linear_layer}"
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
    optimization.lr="${optimization_lr}" \
    task.data=/opt/gpfs/home/chushu/data/audioset/setting/SFT_AS20k \
    task.target_length=1024 \
    task.roll_aug=true \
    task.load_clap_emb=false \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    optimizer.groups.default.lr_float=${optimizer_groups_default_lr_float} \
    model.model_path=${model_model_path} \
    +model.linear_classifier=true \
    +model.linear_layer=${model_linear_layer} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN # CLS_TOKEN num_workers: 6 data_buffer_size: 10