import config
from datetime import datetime
import time
import tensorflow as tf
import itertools
import os
import transformer_model
import csv

from keras.callbacks import EarlyStopping, CSVLogger
from keras.optimizers import Adam


def structure_enumerate(container_structures):
    """
    Creates a generator with all permutation of the dictionary's values.
    The keys of the dictionary will stay intact
    """
    container_values = list(container_structures.values())
    for product_values in itertools.product(*container_values):
        yield dict(zip(container_structures.keys(), product_values))


def structure_name(structure):
    """
    returns a human readable name for the test structure
    """
    return "-".join([str(x) for x in structure.values()])


def structure_tested(structure):
    """
    returns true if the structure has been used for a performance test
    """
    if not os.path.isfile(config.FILE_EXECUTED_TESTS):
        return False
    name = structure_name(structure)
    with open(config.FILE_EXECUTED_TESTS) as f:
        return name in [s.strip() for s in f.readlines()]


def structure_ran_successfully(structure):
    """
    adds the structure name to the list of successfully ran tests
    """
    with open(config.FILE_EXECUTED_TESTS, 'a') as f:
        f.write(structure_name(structure) + "\n")


def save_model_inception(size, count_classes):
    """
    Save & compile the tensorflow inception v3 model
    """
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        input_shape=config.SIZE_IMAGENET_DATA + (3,),
        classes=count_classes,
        classifier_activation='softmax'
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
    model.save(
        config.FILE_BASE_MODEL_INCEPTION.format(size),
        include_optimizer=True)
    model.summary()


def load_model_inception(weights_size):
    """
    Loads the tensorflow inception v3 model
    """

    return tf.keras.models.load_model(config.FILE_BASE_MODEL_INCEPTION.format(weights_size))


def build_model_transformer(input_vocab_size, target_vocab_size):
    """
    Builds the transformer model
    """
    model = transformer_model.Transformer(
            num_layers=transformer_model.NUM_LAYERS,
            d_model=transformer_model.D_MODEL,
            num_heads=transformer_model.NUM_HEADS,
            dff=transformer_model.DFF,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            dropout_rate=transformer_model.DROPOUT_RATE)
    model.build([
        tf.TensorShape([1, config.MAX_TOKENS]),
        tf.TensorShape([1, config.MAX_TOKENS])
    ])
    learning_rate = transformer_model.CustomSchedule(
                transformer_model.D_MODEL)
    optimizerAdam = Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    model.compile(
        optimizer=optimizerAdam,
        loss=transformer_model.masked_loss,
        metrics=[transformer_model.masked_accuracy]
    )
    return model

def save_model_transformer(size, input_vocab_size, target_vocab_size):
    """
    Save & build the tensorflow transformer weights
    """
    model = build_model_transformer(input_vocab_size, target_vocab_size)
    model.save_weights(config.FILE_BASE_MODEL_TRANSFORMER.format(size))
    model.summary()


def load_model_transformer(size, input_vocab_size, target_vocab_size):
    """
    Loads & compile the tensorflow transformer weights
    """
    model = build_model_transformer(input_vocab_size, target_vocab_size)
    model.load_weights(config.FILE_BASE_MODEL_TRANSFORMER.format(size))
    return model


def build_test(structure):
    """
    Resolves the model type which should be used for testing
    Returns an initilalized TestRun object
    """
    if config.K_ALGO in structure:
        if structure[config.K_ALGO] == config.V_ALGO_INCEPTION:
            return InceptionTestRun(structure)
        elif structure[config.K_ALGO] == config.V_ALGO_TRANSFORMER:
            return TransformerTestRun(structure)

    raise NotImplementedError()


class TestRun:
    """
    Abstract class for running test cases
    """

    def __init__(self, structure):
        self.structure = structure
        self.startDateTime = None
        self.endDateTime = None

    def pre(self):
        if not os.path.isdir(config.DIR_RESULTS):
            os.mkdir(config.DIR_RESULTS)
        if not os.path.isdir(self.get_dir_result()):
            os.mkdir(self.get_dir_result())

        for gpu_id in config.GPU_IDS:
            clock_speed = int(self.get_gpu_speed() * config.GPU_MAX_SPEED)
            os.system(
                f'sudo nvidia-smi -i {gpu_id} --lock-gpu-clocks=0,{clock_speed}')

        for pci_slot in config.PCI_SLOTS:
            pci_generation = self.get_pci_generation()
            os.system(f'sudo ./pcie_set_speed.sh {pci_slot} {pci_generation}')

        time.sleep(config.PRE_SLEEP_TIME)

    def post(self):
        for gpu_id in config.GPU_IDS:
            os.system(f'sudo nvidia-smi -i {gpu_id} --reset-gpu-clocks')
        for pci_slot in config.PCI_SLOTS:
            os.system(f'sudo ./pcie_set_speed.sh {pci_slot} 4')


        with open(self.get_file_time(), 'w') as fileTime:
            fileTime.write(self.startDateTime.isoformat())
            fileTime.write("\n")
            fileTime.write(self.endDateTime.isoformat())
            fileTime.write("\n")
            fileTime.write(str(self.endDateTime - self.startDateTime))
        structure_ran_successfully(self.structure)

    def run(self):
        raise NotImplementedError()

    def get_dir_result(self):
        return config.DIR_RESULT_ALGO.format(structure_name(self.structure))

    def get_file_log_fit(self):
        return config.FILE_LOG_FIT.format(self.get_dir_result())

    def get_file_log_evaluate(self):
        return config.FILE_LOG_EVALUATE.format(self.get_dir_result())

    def get_file_time(self):
        return config.FILE_TIME.format(self.get_dir_result())

    def get_file_weights(self):
        return config.FILE_WEIGHTS.format(self.get_dir_result())

    def get_data_size(self):
        return int(self.structure[config.K_DATA_SIZE_PERC])

    def get_batch_size(self):
        return int(self.structure[config.K_BATCH_SIZE])

    def get_pci_generation(self):
        return self.structure[config.K_PCI_GENERATION]

    def get_gpu_speed(self):
        return float(self.structure[config.K_GPU_PERC]) / 100.0


class InceptionTestRun(TestRun):
    """
    Test class for the inception model
    """
    def __init__(self, structure):
        self.mirrored_strategy = tf.distribute.MirroredStrategy(devices=config.LIST_GPUS)
        TestRun.__init__(self, structure)

    def pre(self):
        self.train_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_TRAIN.format(self.get_data_size()),
            batch_size=self.get_batch_size(),
            image_size=config.SIZE_IMAGENET_DATA,
            seed=config.RANDOM_SEED,
            label_mode='categorical'
        ).map(self.preprocess).prefetch(self.get_batch_size() * config.COUNT_PREFETCH_DATA)

        self.val_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_VAL.format(self.get_data_size()),
            batch_size=self.get_batch_size(),
            image_size=config.SIZE_IMAGENET_DATA,
            seed=config.RANDOM_SEED,
            label_mode='categorical'
        ).map(self.preprocess).prefetch(self.get_batch_size() * config.COUNT_PREFETCH_DATA)

        self.test_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_TEST.format(self.get_data_size()),
            image_size=config.SIZE_IMAGENET_DATA,
            seed=config.RANDOM_SEED,
            label_mode='categorical'
        ).map(self.preprocess)

        with self.mirrored_strategy.scope():
            self.model = load_model_inception(self.get_data_size())

        TestRun.pre(self)

    def run(self):
        self.startDateTime = datetime.now()
        self.model.fit(
            self.train_data,
            epochs=config.MAX_EPOCHS,
            validation_data=self.val_data,
            verbose=config.VERBOSE,
            callbacks=[
                EarlyStopping(
                    monitor='loss',
                    patience=config.MAX_PATIENCE_LOSS
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.MAX_PATIENCE_VAL_LOSS
                ),
                CSVLogger(self.get_file_log_fit())
            ]
        )
        self.endDateTime = datetime.now()

    def post(self):
        self.model.save_weights(self.get_file_weights())
        logs = self.model.evaluate(
            self.test_data,
            verbose=config.VERBOSE,
            return_dict=True
        )

        with open(self.get_file_log_evaluate(), 'w') as fileLog:
            writer = csv.DictWriter(fileLog, fieldnames=logs.keys())
            writer.writeheader()
            writer.writerow(logs)

        TestRun.post(self)

    def preprocess(self, image, label):
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image, label


class TransformerTestRun(TestRun):
    """
    Test class for the transformer model
    """
    def __init__(self, structure):
        # works the same as MirroredStrategy on single device but has a better reduce operator for arrays
        self.mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        TestRun.__init__(self, structure)

    def pre(self):
        self.en_tokenizer = transformer_model.CustomTokenizer(
            config.FILE_TRANSFORMER_TOKENIZER_EN.format(self.get_data_size()))
        self.fr_tokenizer = transformer_model.CustomTokenizer(
            config.FILE_TRANSFORMER_TOKENIZER_FR.format(self.get_data_size()))

        self.train_data = tf.data.experimental.make_csv_dataset(
            config.FILE_KAGGLE_ENFR_TRAIN.format(self.get_data_size()),
            ignore_errors=True,
            batch_size=self.get_batch_size(),
            num_epochs=1,
            header=True,
            shuffle=False
        ).map(self.preprocess).prefetch(self.get_batch_size() * config.COUNT_PREFETCH_DATA)

        self.val_data = tf.data.experimental.make_csv_dataset(
            config.FILE_KAGGLE_ENFR_VAL.format(self.get_data_size()),
            ignore_errors=True,
            batch_size=self.get_batch_size(),
            num_epochs=1,
            header=True,
            shuffle=False
        ).map(self.preprocess).prefetch(self.get_batch_size() * config.COUNT_PREFETCH_DATA)

        self.test_data = tf.data.experimental.make_csv_dataset(
            config.FILE_KAGGLE_ENFR_TEST.format(self.get_data_size()),
            ignore_errors=True,
            batch_size=self.get_batch_size(),
            num_epochs=1,
            header=True,
            shuffle=False
        ).map(self.preprocess)

        with self.mirrored_strategy.scope():
            self.model = load_model_transformer(
                self.get_data_size(),
                self.en_tokenizer.get_vocab_size(),
                self.fr_tokenizer.get_vocab_size()
            )

        TestRun.pre(self)

    def run(self):
        self.startDateTime = datetime.now()
        self.model.fit(
            self.train_data,
            epochs=config.MAX_EPOCHS,
            validation_data=self.val_data,
            verbose=config.VERBOSE,
            callbacks=[
                EarlyStopping(
                    monitor='loss',
                    patience=config.MAX_PATIENCE_LOSS
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.MAX_PATIENCE_VAL_LOSS
                ),
                CSVLogger(self.get_file_log_fit())
            ]
        )
        self.endDateTime = datetime.now()

    def post(self):
        self.model.save_weights(self.get_file_weights())
        logs = self.model.evaluate(
            self.test_data,
            verbose=config.VERBOSE,
            return_dict=True
        )

        with open(self.get_file_log_evaluate(), 'w') as fileLog:
            writer = csv.DictWriter(fileLog, fieldnames=logs.keys())
            writer.writeheader()
            writer.writerow(logs)

        TestRun.post(self)

    def preprocess(self, row):
        iteratorColumns = iter(row.values())
        en_string_tensor = next(iteratorColumns)
        fr_string_tensor = next(iteratorColumns)

        en = self.en_tokenizer.tokenize(en_string_tensor)  # Output is ragged.
        en = en[:, :config.MAX_TOKENS]                    # Trim to MAX_TOKENS.
        # Convert to 0-padded dense Tensor
        en = en.to_tensor()

        fr = self.fr_tokenizer.tokenize(fr_string_tensor)
        fr = fr[:, :(config.MAX_TOKENS+1)]
        fr_inputs = fr[:, :-1].to_tensor()  # Drop the [END] tokens
        fr_labels = fr[:, 1:].to_tensor()   # Drop the [START] tokens

        return (en, fr_inputs), fr_labels
