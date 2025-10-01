weight_file=/opt/gpfs/home/chushu/exp/eat/pre_4_AS2M/clap_1_2025-09-08_14-08-54/checkpoint_last.pt
prediction_mode=CLS_TOKEN
lr=0.0003
device=0

rm -rf /opt/gpfs/home/chushu/codes/2506/EAT/evar/logs
rm -rf /opt/gpfs/home/chushu/codes/2506/EAT/evar/results
CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml cremad batch_size=16,weight_file=$weight_file $lr
CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml gtzan batch_size=16,weight_file=$weight_file $lr
# CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml spcv2 batch_size=64,weight_file=$weight_file $lr
CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml esc50 batch_size=64,weight_file=$weight_file $lr
# CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml us8k batch_size=64,weight_file=$weight_file $lr
#CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml vc1 batch_size=64,weight_file=$weight_file $lr
# CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml voxforge batch_size=64,weight_file=$weight_file $lr
# CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml nsynth batch_size=64,weight_file=$weight_file $lr
# CUDA_VISIBLE_DEVICES=${device} python ./evar/2pass_lineareval.py ./evar/config/eat_clap_0.yaml surge batch_size=64,weight_file=$weight_file $lr

CUDA_VISIBLE_DEVICES=${device} python ./evar/summarize.py $weight_file
