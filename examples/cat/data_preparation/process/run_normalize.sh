# opt split: bal_train  eval  unbal_train; all to momo by librosa
cd examples/cat/data_preparation/process
split=eval
python normalize.py \
    --source_dir /inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/audioset/raw_data/${split} \
    --target_dir  /inspire/hdd/global_user/zhouchushu-253108120180/raw_datas/audioset/wav_16k/${split} \
    --sample_rate 16000 \
    --log_file /inspire/hdd/global_user/zhouchushu-253108120180/codes/2506/EAT/EAT/prepapre_data/process/log/normalize_${split}1.1og \
    --num_processes 64