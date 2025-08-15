cd /opt/gpfs/home/chushu/codes/2506/fairseq
pip install --editable ./
pip install timm
pip install tensorboardX
pip install editdistance

export PYTHONPATH="/opt/gpfs/home/chushu/codes/2506/test/fairseq:$PYTHONPATH"
python fairseq_cli/hydra_train.py -m --config-dir examples/wav2vec/config/finetuning --config-name vox_10h \
task.data=/opt/gpfs/home/chushu/data/librispeech model.w2v_path=/opt/gpfs/home/chushu/codes/2506/fairseq/multirun/2025-07-27/14-16-26/0/checkpoints/checkpoint_last.pt common.user_dir=examples/data2vec distributed_training.distributed_world_size=${1:-1}
