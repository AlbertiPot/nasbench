"""
Data:2021/08/26
Target: 
从原始的nasbench101_108eps（tfrecord文件）中提取423624个结构的数据到json方便计算，共有fixed_metrics和computed_metrics两个数据

fixed_metrics
{'module_adjacency': 
      array(
      [[0, 1, 0, 0, 1, 1, 0],
       [0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0]], dtype=int8), 
      
      'module_operations': ['input', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'output'], 
      
      'trainable_parameters': 8555530
}

computed_metrics
{108: 
      [
      {'halfway_training_time': 883.4580078125, 'halfway_train_accuracy': 0.8282251358032227, 'halfway_validation_accuracy': 0.7776442170143127, 'halfway_test_accuracy': 0.7740384340286255, 'final_training_time': 1769.1279296875, 'final_train_accuracy': 1.0, 'final_validation_accuracy': 0.9241786599159241, 'final_test_accuracy': 0.9211738705635071}, 
      {'halfway_training_time': 883.6810302734375, 'halfway_train_accuracy': 0.8796073794364929, 'halfway_validation_accuracy': 0.8291265964508057, 'halfway_test_accuracy': 0.8204126358032227, 'final_training_time': 1768.2509765625, 'final_train_accuracy': 1.0, 'final_validation_accuracy': 0.9245793223381042, 'final_test_accuracy': 0.9190705418586731}, 
      {'halfway_training_time': 883.4569702148438, 'halfway_train_accuracy': 0.8634815812110901, 'halfway_validation_accuracy': 0.811598539352417, 'halfway_test_accuracy': 0.8058894276618958, 'final_training_time': 1768.9759521484375, 'final_train_accuracy': 1.0, 'final_validation_accuracy': 0.9304887652397156, 'final_test_accuracy': 0.9215745329856873}
      ]
}

仅仅保留 邻接矩阵，算子，参数数量，3次平均的train_acc, val_acc, test_acc

借鉴了nasbench的example.py
"""

import numpy as np
import json
import os
import operator
from tqdm import tqdm

def main(dataset_path, save_path):
    from nasbench import api

    nasbench = api.NASBench(dataset_file=dataset_path)
    
    assert len(nasbench.fixed_statistics) == len(nasbench.computed_statistics) == 423624, "Wrong length of the original dataset"
    print('\nIterating over unique models in the dataset.\n')

    dataset_json = []
    for unique_hash in nasbench.hash_iterator():
        
        arch = {}
        fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(unique_hash)
    
        arch.update(fixed_metrics)
        arch["module_adjacency"] = arch["module_adjacency"].tolist()

        single_eps_data = computed_metrics[108]
        assert isinstance(single_eps_data, list)
        assert len(single_eps_data) == 3, "Wrong length of runs"
        
        arch["avg_training_time"] = np.mean([v["final_training_time"] for _,v in enumerate(single_eps_data)])
        arch["avg_train_accuracy"] = np.mean([v["final_train_accuracy"] for _,v in enumerate(single_eps_data)])
        arch["avg_validation_accuracy"] = np.mean([v["final_validation_accuracy"] for _,v in enumerate(single_eps_data)])
        arch["avg_test_accuracy"] = np.mean([v["final_test_accuracy"] for _,v in enumerate(single_eps_data)])
        
        arch["unique_hash"] = unique_hash
        
        dataset_json.append(arch)
    
    assert len(dataset_json) == 423624, "Wrong length of the extracted json dataset"

    with open(save_path, 'w') as f:
        json.dump(dataset_json, f)
    
    print('extract finished')

def compare_json_with_ctns(json_path, ctnas_json_path):
    
    with open(json_path, 'r') as f:
        extracted_json = json.load(f)
    
    with open(ctnas_json_path, 'r') as g:
        ctnas_json = json.load(g)

    extracted_json_length = len(extracted_json)
    ctnas_json_lenght = len(ctnas_json)
    assert extracted_json_length == ctnas_json_lenght == 423624, "the length of the extracted json dataset is not the same with CTNAS dataset"


    extracted_json.sort(key = lambda elem0 : elem0["avg_validation_accuracy"])
    ctnas_json.sort(key = lambda elem1 : elem1["validation_accuracy"])
    
    for i, compond in enumerate(tqdm(zip(extracted_json, ctnas_json))):        
        x,y= compond
        assert operator.eq(x["module_adjacency"], y["matrix"])
        assert operator.eq(x["module_operations"], y["ops"])
        assert x["trainable_parameters"] == y["n_params"]
        assert x["avg_training_time"] == y["training_time"]
        assert x["avg_train_accuracy"] == y["train_accuracy"]
        assert x["avg_validation_accuracy"] == y["validation_accuracy"]
        assert x["avg_test_accuracy"] == y["test_accuracy"]
        assert x["unique_hash"] == y["hash_"]
    print('all ok')


if __name__ == "__main__":
    
    NASBENCH_TFRECORD = '/home/gbc/workspace/nasbench/data/nasbench_only108.tfrecord'
    JSON_SAVE_PATH = '/home/gbc/workspace/nasbench/data/nasbench_only108.json'
    CTNAS_JSON_PATH = '/home/gbc/workspace/CTNAS/ctnas/data/nas_bench.json'
    INPUT = 'input'
    OUTPUT = 'output'
    CONV1X1 = 'conv1x1-bn-relu'
    CONV3X3 = 'conv3x3-bn-relu'
    MAXPOOL3X3 = 'maxpool3x3'

    if os.path.exists(JSON_SAVE_PATH) is False:
        main(dataset_path = NASBENCH_TFRECORD, save_path = JSON_SAVE_PATH)
    if os.path.exists(CTNAS_JSON_PATH) and os.path.exists(JSON_SAVE_PATH):
        compare_json_with_ctns(json_path =JSON_SAVE_PATH, ctnas_json_path=CTNAS_JSON_PATH)

    

