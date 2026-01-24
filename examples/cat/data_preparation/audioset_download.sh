export HF_ENDPOINT="https://hf-mirror.com"

cd examples/cat/data_preparation
python download.py --repo_id confit/audioset-qiuqiangkong --repo_type dataset --local_dir "path to save the dataset" --hf_token "your huggingface token"