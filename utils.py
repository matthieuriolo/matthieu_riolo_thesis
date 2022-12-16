import config
import datetime
import time
import tensorflow as tf
import itertools
import os
import transformer_model

from keras.applications.inception_v3 import preprocess_input


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


def get_model_inception(count_classes, weights_size=None):
    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        input_shape=None,
        classes=count_classes,
        classifier_activation='softmax'
    )

    if weights_size:
        model.load_weights(config.FILE_BASE_MODEL_INCEPTION.format(weights_size))

    return model


def get_model_transformer(input_vocab_size, target_vocab_size, weights_size=None):
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    model = transformer_model.Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout_rate=dropout_rate)

    if weights_size:
        model.load_weights(config.FILE_BASE_MODEL_TRANSFORMER.format(weights_size))

    return model


def build_test(structure):
    if config.K_ALGO in structure:
        if structure[config.K_ALGO] == config.V_ALGO_INCEPTION:
            return InceptionTestRun(structure)
        elif structure[config.K_ALGO] == config.V_ALGO_TRANSFORMER:
            return TransformerTestRun(structure)

    raise NotImplementedError()


class TestRun:
    def __init__(self, structure):
        self.structure = structure

    def pre(self):
        time.sleep(config.PRE_SLEEP_TIME)

    def post(self):
        structure_ran_successfully(self.structure)

    def run(self):
        raise NotImplementedError()

    def get_data_size(self):
        return int(self.structure[config.K_DATA_SIZE_PERC])

    def get_batch_size(self):
        return int(self.structure[config.K_BATCH_SIZE])

    def get_pci_speed(self):
        return float(self.structure[config.K_PCI_PERC]) / 100.0

    def get_gpu_speed(self):
        return float(self.structure[config.K_GPU_PERC]) / 100.0


class InceptionTestRun(TestRun):
    # TODO setting cpu, pci speed
    # TODO tf.get_logger()

    def pre(self):
        self.train_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_TRAIN.format(self.get_data_size()),
            batch_size=self.get_batch_size(),
            image_size=(128, 128),
            shuffle=False,
            preprocessing_function=preprocess_input
        )

        self.val_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_VAL.format(self.get_data_size()),
            batch_size=self.get_batch_size(),
            image_size=(128, 128),
            shuffle=False,
            preprocessing_function=preprocess_input
        )

        self.test_data = tf.keras.utils.image_dataset_from_directory(
            config.DIR_IMAGENET_TEST.format(self.get_data_size()),
            image_size=(128, 128),
            shuffle=False,
            preprocessing_function=preprocess_input
        )

        self.model = get_model_inception(
            self.train_data.class_names, self.get_data_size())
        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        TestRun.pre(self)

    def run(self):
        self.startDateTime = datetime.now()
        self.model.fit(
            self.train_data,
            epochs=config.MAX_EPOCHS,
            validation_data=self.val_data,
            verbose=2
        )
        self.endDateTime = datetime.now()

    def post(self):
        with open(config.FILE_IMAGENET_TIME.format(self.get_data_size()), 'w') as fileTime:
            fileTime.write(self.startDateTime.isoformat())
            fileTime.write(self.endDateTime.isoformat())
            fileTime.write(str(self.endDateTime - self.startDateTime))

        self.model.save_weights(
            config.FILE_IMAGENET_WEIGHTS.format(self.get_data_size()))

        # TODO save result from test set
        self.model.evaluate(self.test_data, verbose=2)
        TestRun.post(self)


class TransformerTestRun(TestRun):
    pass
