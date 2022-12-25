#!/usr/bin/env python3

import config
import utils
import os
import shutil
import pandas as pd
from tqdm import tqdm
import random
import tensorflow as tf
import tensorflow_text as tftext
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import transformer_model

# helper functions
def split_csv_data(data):
    trainSize = int(config.V_TRAIN_VALIDATION_PERC[0] / 100.0 * len(data.index))
    valSize = int(config.V_TRAIN_VALIDATION_PERC[1] / 100.0 * len(data.index))
    
    dataTrain = data.iloc[:trainSize,:]
    dataVal = data.iloc[trainSize:(trainSize+valSize),:]
    dataTest = data.iloc[(trainSize+valSize):,:]
    
    return (dataTrain, dataVal, dataTest)

def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)



random.seed(config.RANDOM_SEED)

# cleanup DATA directory
print("Clean up data directory")
if os.path.isdir(config.DIR_DATA_SET):
    shutil.rmtree(config.DIR_DATA_SET)
os.mkdir(config.DIR_DATA_SET)


# download ImageNet DataSet
print("Download imagenet train dataset")
os.system(f'kaggle datasets download -d j53t3r/imagenet128x128 -p {config.DIR_IMAGENET_SET}')
os.system(f'unzip -d {config.DIR_IMAGENET_SET} {config.DIR_IMAGENET_SET}/imagenet128x128.zip')

# merge the images locations together
print("Merge all imagenet dataset together")
os.mkdir(config.DIR_IMAGENET_DATA)
os.system(f'cp -R {config.DIR_IMAGENET_SET}/Imagenet128x128/train_data/box/* {config.DIR_IMAGENET_DATA}/')
os.system(f'cp -R {config.DIR_IMAGENET_SET}/Imagenet128x128/val_data/box/* {config.DIR_IMAGENET_DATA}/')

# build different dataset sizes
print("Create different imagenet dataset sizes")
imageNetClasses = [dir for dir in os.listdir(config.DIR_IMAGENET_DATA)
                    if not dir.startswith('.')
                    and os.path.isdir(config.DIR_IMAGENET_DATA + '/' + dir)]
random.shuffle(imageNetClasses)

for size in config.V_DATA_SIZE_PERC:
    print("- ImageNet class size: {}".format(size))
    perc_classes = size / 100.0
    dataClasses = imageNetClasses[:int(len(imageNetClasses) * perc_classes)]

    dirTrain = config.DIR_IMAGENET_TRAIN.format(size)
    dirVal = config.DIR_IMAGENET_VAL.format(size)
    dirTest = config.DIR_IMAGENET_TEST.format(size)
    os.mkdir(dirTrain)
    os.mkdir(dirVal)
    os.mkdir(dirTest)

    for dataClass in  tqdm(dataClasses):
        os.system(f'cp -R {config.DIR_IMAGENET_DATA}/{dataClass} {dirTrain}/')
        dataImgs = [img for img in os.listdir(f'{dirTrain}/{dataClass}')
                                    if os.path.isfile(f'{dirTrain}/{dataClass}/' + img)]
        
        random.shuffle(dataImgs)

        trainSize = int(config.V_TRAIN_VALIDATION_PERC[0] / 100.0 * len(dataImgs))
        valSize = int(config.V_TRAIN_VALIDATION_PERC[1] / 100.0 * len(dataImgs))
        
        os.mkdir(f'{dirVal}/{dataClass}')
        os.mkdir(f'{dirTest}/{dataClass}')

        valData = dataImgs[trainSize:trainSize+valSize]
        testData = dataImgs[trainSize+valSize:]
        
        for imgValPath in valData:
            shutil.move(f'{config.DIR_IMAGENET_DATA}/{dataClass}/{imgValPath}', f'{dirVal}/{dataClass}')
        for imgTestPath in testData:
            shutil.move(f'{config.DIR_IMAGENET_DATA}/{dataClass}/{imgTestPath}', f'{dirTest}/{dataClass}')

# download kaggle https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset
print("Download and unzip kaggle en/fr translation dataset")
os.system(f'kaggle datasets download -d dhruvildave/en-fr-translation-dataset -p {config.DIR_KAGGLE_ENFR_SET}')
os.system(f'unzip -d {config.DIR_KAGGLE_ENFR_SET} {config.DIR_KAGGLE_ENFR_SET}/en-fr-translation-dataset.zip')

# split kaggle data set
print("Split kaggle en/fr translation dataset")
kaggleData = pd.read_csv(config.DIR_KAGGLE_ENFR_SET + '/en-fr.csv')
kaggleData = kaggleData.sample(frac = 1, random_state=config.RANDOM_SEED)
for size in config.V_DATA_SIZE_PERC:
    data = kaggleData.iloc[:int(size / 100.0 * len(kaggleData.index)),:]
    dataTrain, dataVal, dataTest = split_csv_data(data)

    dataTrain.to_csv(config.FILE_KAGGLE_ENFR_TRAIN.format(size), index = False)
    dataVal.to_csv(config.FILE_KAGGLE_ENFR_VAL.format(size), index = False)
    dataTest.to_csv(config.FILE_KAGGLE_ENFR_TEST.format(size), index = False)

# build inception from tensorflow application
print("Create base inceptionv3 models")
if not os.path.isdir(config.DIR_BASE_MODEL_INCEPTION):
    os.mkdir(config.DIR_BASE_MODEL_INCEPTION)

for size in config.V_DATA_SIZE_PERC:
    train_data = tf.keras.utils.image_dataset_from_directory(
        config.DIR_IMAGENET_TRAIN.format(size),
        shuffle=False,
    )
    count_classes = len(train_data.class_names)
    model = utils.get_model_inception(count_classes)
    model.save_weights(config.FILE_BASE_MODEL_INCEPTION.format(size))


# build transformer model & tokens
print("Create tokenizer data for transformer")
if not os.path.isdir(config.DIR_BASE_MODEL_TRANSFORMER):
    os.mkdir(config.DIR_BASE_MODEL_TRANSFORMER)

for size in config.V_DATA_SIZE_PERC:
    train_set = tf.data.experimental.make_csv_dataset(
        config.FILE_KAGGLE_ENFR_TRAIN.format(size),
        ignore_errors=True,
        batch_size=1000,
        num_epochs=1,
        header=True,
        shuffle=False
    )
    
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = 10000000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"],
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=config.BERT_TOKENIZER_PARAMS,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    train_en = train_set.map(lambda row: row['en'])
    en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en,
        **bert_vocab_args
    )
    
    train_fr = train_set.map(lambda row: row['fr'])
    fr_vocab = bert_vocab.bert_vocab_from_dataset(
        train_fr,
        **bert_vocab_args
    )
    write_vocab_file(config.FILE_TRANSFORMER_TOKENIZER_EN.format(size), en_vocab)
    write_vocab_file(config.FILE_TRANSFORMER_TOKENIZER_FR.format(size), fr_vocab)

print("Create base transformer model")
for size in config.V_DATA_SIZE_PERC:
    en_tokenizer = transformer_model.CustomTokenizer(config.FILE_TRANSFORMER_TOKENIZER_EN.format(size))
    fr_tokenizer = transformer_model.CustomTokenizer(config.FILE_TRANSFORMER_TOKENIZER_FR.format(size))
    model = utils.get_model_transformer(en_tokenizer.get_vocab_size(), fr_tokenizer.get_vocab_size())
    model.save_weights(config.FILE_BASE_MODEL_TRANSFORMER.format(size))