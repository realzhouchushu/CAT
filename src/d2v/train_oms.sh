cd /opt/gpfs/home/chushu/codes/2506/fairseq
pip install --editable ./
pip install timm

export PYTHONPATH="/opt/gpfs/home/chushu/codes/2506/fairseq:$PYTHONPATH"
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task task.data=/opt/gpfs/home/chushu/data/librispeech distributed_training.distributed_world_size=${1:-1}
# python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
# --config-name base_audio_only_task task.data=/opt/gpfs/home/chushu/data/librispeech optimization.max_update=400000 +lr_scheduler.max_update=400000 distributed_training.distributed_world_size=${1:-1}