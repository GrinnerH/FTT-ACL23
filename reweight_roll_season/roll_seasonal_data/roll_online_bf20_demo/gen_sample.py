import json
import random

# 读取JSON文件
for index in range(1,5):
    path = 'reweight_roll_season/roll_seasonal_data/roll_online_bf20_demo/roll_season_' + f"{index}" + '/train.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机抽取100个条目
    sampled_data = random.sample(data, 100)

    # 保存为新的JSON文件
    save_path = 'reweight_roll_season/roll_seasonal_data/roll_online_bf20_demo/roll_season_' + f"{index}" + '/sampled_train.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)
