# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-26 19:15
LastEditTime: 2023-03-01 22:25
LastEditors: MrZQAQ
Description: main file of project
FilePath: /MCANet/main.py
'''

import argparse

from RunModel import run_model, ensemble_run_model
from model import MCANet, onlyPolyLoss

parser = argparse.ArgumentParser(
    prog='MCANet',
    description='MCANet is model in paper: \"MultiheadCrossAttention based network model for DTI prediction\"',
    epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=[
                    "DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['MCANet', 'MCANet-B', 'onlyPolyLoss', 'onlyMCA'],
                    default='MCANet', help='Which model to use, \"MCANet\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
args = parser.parse_args()

if args.model == 'MCANet':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=MCANet, K_Fold=args.fold, LOSS='PolyLoss')
if args.model == 'onlyMCA':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=MCANet, K_Fold=args.fold, LOSS='CrossEntropy')
if args.model == 'onlyPolyLoss':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=onlyPolyLoss, K_Fold=args.fold, LOSS='PolyLoss')
if args.model == 'MCANet-B':
    ensemble_run_model(SEED=args.seed, DATASET=args.dataSetName, K_Fold=args.fold)
