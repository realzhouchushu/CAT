cd /opt/gpfs/home/chushu/codes/2506/fairseq
pip install nvitop
pip install --editable ./

python fairseq_cli/hydra_train.py -m \
    --config-dir EAT/config \
    --config-name finetuning  \
    common.user_dir=EAT \
    checkpoint.save_dir=/opt/gpfs/home/chushu/exp/eat/sft_8_AS20k \
    checkpoint.restore_file=/opt/gpfs/home/chushu/exp/eat/sft_8_AS20k/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    distributed_training.distributed_world_size=${1:-1} \
    dataset.num_workers=32 \
    dataset.batch_size=48 \
    task.data=/opt/gpfs/home/chushu/data/AudioSet/SFT_AS20K \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=/opt/gpfs/home/chushu/exp/eat/pre_8_AS2M/checkpoint_last.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN