export PYTHONPATH="/opt/gpfs/home/chushu/codes/2506/EAT:$PYTHONPATH"
python fairseq_cli/hydra_train.py -m --config-dir examples/wav2vec/config/finetuning --config-name vox_10h \
task.data=/opt/gpfs/home/chushu/data/librispeech model.w2v_path=/opt/gpfs/home/chushu/codes/2506/EAT/multirun/2025-07-27/14-16-26/0/checkpoints/checkpoint_last.pt common.user_dir=examples/data2vec distributed_training.distributed_world_size=1
