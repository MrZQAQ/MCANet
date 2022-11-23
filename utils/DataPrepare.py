# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-29 13:59
LastEditTime: 2022-11-23 15:33
LastEditors: MrZQAQ
Description: Prepare Data for main process
FilePath: /MCANet/utils/DataPrepare.py
CopyRight 2022 by MrZQAQ. All rights reserved.
'''

import numpy as np

def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset
