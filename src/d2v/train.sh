export PYTHONPATH="/opt/gpfs/home/chushu/codes/2506/EAT:$PYTHONPATH"
python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \
--config-name base_audio_only_task task.data=/opt/gpfs/home/chushu/data/librispeech optimization.max_update=4000000 +lr_scheduler.max_update=4000000 distributed_training.distributed_world_size=1