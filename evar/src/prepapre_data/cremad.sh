# 原始总量7438， 可获取总量7429
python ./evar/evar/utils/download_cremad.py /opt/gpfs/data/raw_data/cremad/raw_audio
python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/cremad/raw_audio /opt/gpfs/data/raw_data/cremad/16k/ 16000
python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/cremad/raw_audio /opt/gpfs/data/raw_data/cremad/22k/ 22000
python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/cremad/raw_audio /opt/gpfs/data/raw_data/cremad/32k/ 32000
python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/cremad/raw_audio /opt/gpfs/data/raw_data/cremad/44k/ 44100
python ./evar/prepare_wav.py /opt/gpfs/data/raw_data/cremad/raw_audio /opt/gpfs/data/raw_data/cremad/48k/ 48000