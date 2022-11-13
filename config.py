#!/usr/bin/env python3

import itertools
from itertools import permutations
import os

# file locations
FILE_EXECUTED_TESTS = 'executed_tests.txt'
DIR_DATA_SET = '/media/matthieuriolo/Backup/matthieu_riolo_thesis/data'
DIR_OPENIMAGE_SET = DIR_DATA_SET + '/openimage'
DIR_OPENIMAGE_TRAIN = DIR_OPENIMAGE_SET + '/train'
DIR_OPENIMAGE_VAL = DIR_OPENIMAGE_SET + '/validation'
DIR_OPENIMAGE_TEST = DIR_OPENIMAGE_SET + '/test'

DIR_KAGGLE_ENFR_SET = DIR_DATA_SET + '/kaggle_english_french'
FILE_KAGGLE_ENFR_TRAIN = DIR_KAGGLE_ENFR_SET + '/kaggle_en_fr_train.csv'
FILE_KAGGLE_ENFR_VAL = DIR_KAGGLE_ENFR_SET + '/kaggle_en_fr_validation.csv'
FILE_KAGGLE_ENFR_TEST = DIR_KAGGLE_ENFR_SET + '/kaggle_en_fr_test.csv'


# test configs
PRE_SLEEP_TIME = 60 * 20 # 20min
BASE_MODEL_INCEPTION = DIR_DATA_SET + '/base_inception'
BASE_MODEL_BERT = DIR_DATA_SET + '/base_bert'

# define the different test cases
K_ALGO = 'algorithm'
V_ALGO_BERT = 'bert'
V_ALGO_INCEPTION = 'inception'

K_PCI_PERC = 'pci'
K_GPU_PERC = 'gpu'
K_DATA_SIZE_PERC = 'data_size'
K_COUNT_RUNS = 'count_runs'
K_BATCH_SIZE = 'batch_size'

CONTAINER_STRUCTURES = dict([
    (K_ALGO, [V_ALGO_INCEPTION, V_ALGO_BERT]),
    (K_PCI_PERC, [33, 66, 100]),
    (K_GPU_PERC, [33, 66, 100]),
    (K_DATA_SIZE_PERC, [2, 5, 10, 50, 100]),
    (K_DATA_SIZE_PERC, [64, 128, 256]),
    (K_COUNT_RUNS, [1, 2])
])
