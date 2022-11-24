#!/usr/bin/env python3

import config
import utils
import os
import shutil
import pandas as pd
from tqdm import tqdm

# helper functions

def split_csv_data(data):
    trainSize = int(config.V_TRAIN_VALIDATION_PERC[0] / 100.0 * len(data.index))
    valSize = int(config.V_TRAIN_VALIDATION_PERC[1] / 100.0 * len(data.index))
    testSize = len(data.index) - trainSize - valSize
    dataTrain = data.iloc[:trainSize,:]
    dataVal = data.iloc[trainSize:(trainSize+valSize),:]
    dataTest = data.iloc[(trainSize+valSize):,:]
    return (dataTrain, dataVal, dataTest)


# cleanup DATA directory
# print("Clean up data directory")
# if os.path.isdir(config.DIR_DATA_SET):
#     shutil.rmtree(config.DIR_DATA_SET)
# os.mkdir(config.DIR_DATA_SET)


# # download OpenImage DataSet
# print("Download open images train dataset")
# os.system(f'aws s3 --no-sign-request sync s3://open-images-dataset/train {config.DIR_OPENIMAGE_DATA}')
# print("Download open images validation dataset")
# os.system(f'aws s3 --no-sign-request sync s3://open-images-dataset/validation {config.DIR_OPENIMAGE_DATA}')
# print("Download open images test dataset")
# os.system(f'aws s3 --no-sign-request sync s3://open-images-dataset/test {config.DIR_OPENIMAGE_DATA}')

# print("Download open images train+validation+train+classes labels")
# os.system(f'wget https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv -O {config.DIR_OPENIMAGE_SET}/train.csv')
# os.system(f'wget https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv -O {config.DIR_OPENIMAGE_SET}/val.csv')
# os.system(f'wget https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv -O {config.DIR_OPENIMAGE_SET}/test.csv')
# os.system(f'wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv -O {config.DIR_OPENIMAGE_SET}/classes.csv')

# merge the csv files together
# print("Merge all open images dataset together")
# labelsTrainData = pd.read_csv(f'{config.DIR_OPENIMAGE_SET}/train.csv')
# labelsValData = pd.read_csv(f'{config.DIR_OPENIMAGE_SET}/val.csv')
# labelsTestData = pd.read_csv(f'{config.DIR_OPENIMAGE_SET}/test.csv')

# labelsAllData = pd.concat([labelsTrainData, labelsValData, labelsTestData], ignore_index=True)
# labelsAllData.to_csv(f'{config.DIR_OPENIMAGE_SET}/all.csv', index=False)

# build different dataset sizes
print("Create different open images dataset sizes")
classData = pd.read_csv(f'{config.DIR_OPENIMAGE_SET}/classes.csv')
classData = classData.sample(frac = 1)

labelsAllData = pd.read_csv(f'{config.DIR_OPENIMAGE_SET}/all.csv')
labelsAllData = labelsAllData[labelsAllData['Confidence'] == 1.0]
labelsAllData = labelsAllData.groupby("ImageID")['LabelName'].apply(list).reset_index(name='LabelName')

for size in config.V_DATA_SIZE_PERC:
    print("- OpenImage size: {}".format(size))
    labels = classData.iloc[:int(size / 100.0 * len(classData.index)),:]
    labels = set(labels['LabelName'].values)
    data = pd.DataFrame(columns = [config.COL_OPENIMAGE_ID, config.COL_OPENIMAGE_LABEL])
    for _, row in tqdm(labelsAllData.iterrows()):
        img_name = row['ImageID'] + '.jpg'
        src = f'{config.DIR_OPENIMAGE_DATA}/{img_name}'
        if not os.path.exists(src):
            continue
        # if size == 100 or set(row['LabelName']).issubset(labels):
        #     data.loc[len(data.index)] = [row['ImageID'], "$".join(row['LabelName'])]
        intersecs = set(row['LabelName']).intersection(labels)
        if len(intersecs):
            data.loc[len(data.index)] = [row['ImageID'], "$".join(intersecs)]
    
    print("- OpenImage subsize: {}".format(len(data)))
    data = data.sample(frac = 1)
    dataTrain, dataVal, dataTest = split_csv_data(data)

    dataTrain.to_csv(config.DIR_OPENIMAGE_TRAIN.format(size), index=False)
    dataVal.to_csv(config.DIR_OPENIMAGE_VAL.format(size), index=False)
    dataTest.to_csv(config.DIR_OPENIMAGE_TEST.format(size), index=False)


# download kaggle https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset
print("Download and unzip kaggle en/fr translation dataset")
os.system(f'kaggle datasets download -d dhruvildave/en-fr-translation-dataset -p {config.DIR_KAGGLE_ENFR_SET}')
os.system(f'unzip -d {config.DIR_KAGGLE_ENFR_SET} {config.DIR_KAGGLE_ENFR_SET}/en-fr-translation-dataset.zip')

# split kaggle data set
print("Split kaggle en/fr translation dataset")
kaggleData = pd.read_csv(config.DIR_KAGGLE_ENFR_SET + '/en-fr.csv')
kaggleData = kaggleData.sample(frac = 1)

for size in config.V_DATA_SIZE_PERC:
    data = kaggleData.iloc[:int(size / 100.0 * len(kaggleData.index)),:]
    dataTrain, dataVal, dataTest = split_csv_data(data)

    dataTrain.to_csv(config.FILE_KAGGLE_ENFR_TRAIN.format(size))
    dataVal.to_csv(config.FILE_KAGGLE_ENFR_VAL.format(size))
    dataTest.to_csv(config.FILE_KAGGLE_ENFR_TEST.format(size))


# def reset_weights(model):
#     """
#     resets the weights of a model
#     Special thanks to https://github.com/keras-team/keras/issues/341
#     """
#     for layer in model.layers:
#         if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
#             reset_weights(layer) #apply function recursively
#             continue

#         #where are the initializers?
#         if hasattr(layer, 'cell'):
#             init_container = layer.cell
#         else:
#             init_container = layer

#         for key, initializer in init_container.__dict__.items():
#             if "initializer" not in key: #is this item an initializer?
#                   continue #if no, skip it

#             # find the corresponding variable, like the kernel or the bias
#             if key == 'recurrent_initializer': #special case check
#                 var = getattr(init_container, 'recurrent_kernel')
#             else:
#                 var = getattr(init_container, key.replace("_initializer", ""))
            
#             if var is not None:
#                 var.assign(initializer(var.shape, var.dtype))

#             var.assign(initializer(var.shape, var.dtype)) # use the initializer


# build inception from tensorflow application
#model = utils.get_model_inception(???)
#model.save_weight(config.BASE_MODEL_INCEPTION)

# build bert from tensorflow hub

# https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1

# model = tfhub.load('https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1')
# model.save_weight(config.BASE_MODEL_BERT)

#model = utils.get_model_bert(???)
#model.save_weight(config.BASE_MODEL_BERT)