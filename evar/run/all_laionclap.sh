NAME=LAIONCLAP
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml cremad batch_size=16,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml gtzan batch_size=16,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml spcv2 batch_size=64,name=$NAME
CUDA_VISIBLE_DEVICES=0 python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml esc50 batch_size=64,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml us8k batch_size=64,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml vc1 batch_size=64,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml voxforge batch_size=64,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml nsynth batch_size=64,name=$NAME
# python ./evar/2pass_lineareval.py ./evar/config/laionclap.yaml surge batch_size=64,name=$NAME

# python ./evar/zeroshot.py ./evar/config/laionclap.yaml cremad batch_size=16,name=$NAME
# python ./evar/zeroshot.py ./evar/config/laionclap.yaml gtzan batch_size=16,name=$NAME
# python ./evar/zeroshot.py ./evar/config/laionclap.yaml nsynth batch_size=64,name=$NAME
# python ./evar/zeroshot.py ./evar/config/laionclap.yaml esc50 batch_size=64,name=$NAME
# python ./evar/zeroshot.py ./evar/config/laionclap.yaml us8k batch_size=64,name=$NAME

python ./evar/summarize.py $NAME
