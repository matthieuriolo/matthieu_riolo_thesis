# Matthieu Riolo Thesis

This repository holds the code which are used to create the perfomance test.

## Installation

Create at first the virtualenv environment

```
pip3 install virtualenv
virtualenv env
source env/bin/activate
```

Some of the scripts need root permissions. They can be configured by copying the file sudoers into /etc/sudoers.d/

```
cp ./sudoers /etc/sudoers.d/
```

The preparation script will download data from kaggle. You need to configure your system with an authentication token from kaggle. You must follow this instruction:
https://www.kaggle.com/docs/api#getting-started-installation-&-authentication


## Configurations

All configurations can be found in config.py. Adjust them according to your needs. Mainly the download path (DIR_DATA_SET) for the kaggle datasets needs to be configured. Keep in mind that you will need 200GB of storage for the kaggle data itself.

## Preparation

Run the preparetion script with

```
python prepare.py
```

This can take up to 2 days!

## Execution

The test cases can be executed with

```
XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda python run_tests.py
```

There is a delay between every run of PRE_SLEEP_TIME (default 10min) seconds. The resulting models, duration and validation score will be stored in DIR_RESULTS. If a test case runs successfully then an entry is stored in FILE_EXECUTED_TESTS. Rerunning the script wont execute the test a second time if the entry is stored in the above file.