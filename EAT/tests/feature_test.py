import  numpy as np
import json
from tqdm import tqdm

data_path = "/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_CLAP/train.json"
data_path2 = "/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_in/train.json"
data_path3 = "/opt/gpfs/home/chushu/data/audioset/setting/PRETRAIN_AS2M_w_AST/mlp_head_out/train.json"


data = json.load(open(data_path, "r"))
data2 = json.load(open(data_path2, "r"))
data3 = json.load(open(data_path3, "r"))

data_sum = []
data_mean = []
for i in tqdm(range(len(data))):
    clap_path = data[i]["clap_path"]
    a = np.load(clap_path)
    data_sum.append(a.sum())
    data_mean.append(a.mean())
    if i > 10000:
        break

data_sum2 = []
data_mean2 = []
for i in tqdm(range(len(data2))):
    a = np.load(data2[i]["clap_path"])
    data_sum2.append(a.sum())
    data_mean2.append(a.mean())
    if i > 10000:
        break

data_sum3 = []
data_mean3 = []
for i in tqdm(range(len(data3))):
    a = np.load(data3[i]["clap_path"])
    data_sum3.append(a.sum())
    data_mean3.append(a.mean())
    if i > 10000:
        break

if __name__ == "__main__":
    print(np.mean(data_sum))
    print(np.mean(data_mean))
    print(np.mean(data_sum2))
    print(np.mean(data_mean2))
    print(np.mean(data_sum3))
    print(np.mean(data_mean3))