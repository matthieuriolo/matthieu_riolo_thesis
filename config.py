#!/usr/bin/env python3

import math

# data locations
DIR_DATA_SET = '/media/matthieuriolo/WinSSD/code/matthieu_riolo_thesis/data'

DIR_IMAGENET_SET = DIR_DATA_SET + '/imagenet'
DIR_IMAGENET_DATA = DIR_IMAGENET_SET + '/images'
DIR_IMAGENET_TRAIN = DIR_IMAGENET_SET + '/{0}_train'
DIR_IMAGENET_VAL = DIR_IMAGENET_SET + '/{0}_validation'
DIR_IMAGENET_TEST = DIR_IMAGENET_SET + '/{0}_test'

SIZE_KAGGLE_ENFR_MAX_LENGTH = 30
SIZE_KAGGLE_ENFR_MIN_LENGTH = 4
REPLACE_KAGGLE_SYMBOLS = {
    '◦': '',
    r'\u2028': ' ', # line separator
    r'\u0092': "'", # PRIVATE USE TWO
}
SKIP_KAGGLE_CONTAINS_CASE_INSENSITIVE = [
    '\|', # formatting symbol
    '\&',
    '\d', # remove all numbers
    '\(|\)|\[|\]', # removing brackets
    '%', # removing percentage
    '\/', # remove lines with alternative words like TPS/TVH
    ':',
    ';',
    '«',
    '»',
    '>',
    '<',
    '\"',
    '\'',
    '\W\s*[-_]',
    '=',
    '\s+[\?\!\.\,]',
    '\.\s*\.', # remove .. and variants with space between them
    '……',

    # urls
    'http:\/\/'
    'https:\/\/',
    'www\.',
    '\.com',

    # contact adress & emails
    '@',
    'Fax',
    'Tel\.',
    'Telephone:',
    'Tél\.',
    'Téléphone:',
    # weirds&letter indicating bad formatting
    'Appendix',
    '©',
    'Ã',
    '\s\s'
]
SKIP_KAGGLE_CONTAINS_CASE_SENSITIVE = [
    '[A-Z]+\.', # remove titles likes MD. DENT. 
    '[A-Z][A-Z]+', # remove capital words
]
DIR_KAGGLE_ENFR_SET = DIR_DATA_SET + '/kaggle_english_french'
FILE_KAGGLE_ENFR_TRAIN = DIR_KAGGLE_ENFR_SET + '/{0}_train.csv'
FILE_KAGGLE_ENFR_VAL = DIR_KAGGLE_ENFR_SET + '/{0}_validation.csv'
FILE_KAGGLE_ENFR_TEST = DIR_KAGGLE_ENFR_SET + '/{0}_test.csv'
DIR_KAGGLE_ENFR_RESULTS = DIR_KAGGLE_ENFR_SET + '/results/{0}'

DIR_BASE_MODEL_INCEPTION = DIR_IMAGENET_SET + '/base_models'
FILE_BASE_MODEL_INCEPTION = DIR_BASE_MODEL_INCEPTION + '/{0}_inception.h5'

DIR_BASE_MODEL_TRANSFORMER = DIR_KAGGLE_ENFR_SET + '/base_models'
FILE_BASE_MODEL_TRANSFORMER = DIR_BASE_MODEL_TRANSFORMER + '/{0}_transformer.h5'
FILE_TRANSFORMER_TOKENIZER_EN = DIR_BASE_MODEL_TRANSFORMER + '/{0}_en_tokens.txt'
FILE_TRANSFORMER_TOKENIZER_FR = DIR_BASE_MODEL_TRANSFORMER + '/{0}_fr_tokens.txt'


# test configs
GPU_IDS = [0, 1]
LIST_GPUS = ["/gpu:" + str(gpi_id) for gpi_id in GPU_IDS]
GPU_MAX_SPEED = 1837
PCI_SLOTS = ['01:00.0', '04:00.0']
MAX_EPOCHS = 60
MAX_PATIENCE_LOSS = 3
MAX_PATIENCE_VAL_LOSS = math.ceil(MAX_EPOCHS * 0.1) # 10% of epochs
PRE_SLEEP_TIME = 60 * 10 # 10min
VERBOSE = 1

FILE_EXECUTED_TESTS = '/media/matthieuriolo/datascience/matthieu_riolo_thesis/executed_tests.txt'
DIR_RESULTS = '/media/matthieuriolo/datascience/matthieu_riolo_thesis/results'
DIR_RESULT_ALGO = '/media/matthieuriolo/datascience/matthieu_riolo_thesis/results/{0}'
FILE_TIME = '{0}/time.txt'
FILE_WEIGHTS = '{0}/weights.h5'
FILE_LOG_FIT = '{0}/log_fit.txt'
FILE_LOG_EVALUATE = '{0}/log_evaluate.txt'

SIZE_IMAGENET_DATA = (128, 128)


# define the different test cases
K_ALGO = 'algorithm'
V_ALGO_TRANSFORMER = 'transformer'
V_ALGO_INCEPTION = 'inception'

V_BATCH_SIZES = [32, 64, 128]
V_DATA_SIZE_PERC = [2, 5, 10]
V_TRAIN_VALIDATION_PERC = (92, 2) # remaining is used as the size of the test data set

K_PCI_GENERATION = 'pci'
K_GPU_PERC = 'gpu'
K_DATA_SIZE_PERC = 'data_size'
K_COUNT_RUNS = 'count_runs'
K_BATCH_SIZE = 'batch_size'

CONTAINER_STRUCTURES = dict([
    (K_ALGO, [V_ALGO_INCEPTION, V_ALGO_TRANSFORMER]),
    (K_PCI_GENERATION, [1, 2]),
    (K_GPU_PERC, [33, 66, 100]),
    (K_DATA_SIZE_PERC, V_DATA_SIZE_PERC),
    (K_BATCH_SIZE, V_BATCH_SIZES),
    (K_COUNT_RUNS, [1])
])


# others
RANDOM_SEED = 123456
PANDA_CHUNK_SIZE = 10000
BERT_TOKENIZER_PARAMS=dict(lower_case=True)
MAX_TOKENS = SIZE_KAGGLE_ENFR_MAX_LENGTH * 2
