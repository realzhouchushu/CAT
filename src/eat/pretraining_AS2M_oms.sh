cd /opt/gpfs/home/chushu/codes/2506/fairseq
pip install nvitop
pip install --editable ./

python fairseq_cli/hydra_train.py -m \
    --config-dir ./EAT/config \
    --config-name pretraining_AS2M \
    common.user_dir=./EAT \
    checkpoint.save_dir=/opt/gpfs/home/chushu/exp/eat \
    checkpoint.restore_file=/opt/gpfs/home/chushu/exp/eat/checkpoint_last.pt \
    distributed_training.distributed_world_size=${1:-1} \
    dataset.num_workers=24 \
    dataset.batch_size=16 \
    task.data=/opt/gpfs/home/chushu/data/AudioSet/ \
    task.h5_format=False