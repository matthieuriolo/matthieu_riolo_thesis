#!/usr/bin/env python3

import itertools
from itertools import permutations
import os

# file locations
FILE_EXECUTED_TESTS = 'executed_tests.txt'
DIR_DATA_SET = '/media/matthieuriolo/datascience/matthieu_riolo_thesis/data'
DIR_OPENIMAGE_SET = DIR_DATA_SET + '/openimage'
DIR_OPENIMAGE_UNLABELLED = DIR_OPENIMAGE_SET + '/unlabelled'
DIR_OPENIMAGE_LABELLED = DIR_OPENIMAGE_SET + '/labelled'
DIR_OPENIMAGE_TRAIN = DIR_OPENIMAGE_SET + '/{0}_train'
DIR_OPENIMAGE_VAL = DIR_OPENIMAGE_SET + '/{0}_validation'
DIR_OPENIMAGE_TEST = DIR_OPENIMAGE_SET + '/{0}_test'

DIR_KAGGLE_ENFR_SET = DIR_DATA_SET + '/kaggle_english_french'
FILE_KAGGLE_ENFR_TRAIN = DIR_KAGGLE_ENFR_SET + '/{0}_train.csv'
FILE_KAGGLE_ENFR_VAL = DIR_KAGGLE_ENFR_SET + '/{0}_validation.csv'
FILE_KAGGLE_ENFR_TEST = DIR_KAGGLE_ENFR_SET + '/{0}_test.csv'


# test configs
PRE_SLEEP_TIME = 60 * 20 # 20min
BASE_MODEL_INCEPTION = DIR_DATA_SET + '/base_inception'
BASE_MODEL_BERT = DIR_DATA_SET + '/base_bert'

# define the different test cases
K_ALGO = 'algorithm'
V_ALGO_BERT = 'bert'
V_ALGO_INCEPTION = 'inception'

V_BATCH_SIZES = [64, 128, 256]
V_DATA_SIZE_PERC = [2, 5, 10, 50, 100]
V_TRAIN_VALIDATION_PERC = (92, 2) # remaining is used as the size of the test data set

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
    (K_BATCH_SIZE, V_BATCH_SIZES),
    (K_COUNT_RUNS, [1, 2])
])
