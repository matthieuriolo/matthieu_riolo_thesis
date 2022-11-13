import config
import time

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
    if not os.path.isfile(FILE_EXECUTED_TESTS):
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

def get_model_inception(count_classes):
    return tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights=None,
        input_shape=None,
        classes=count_classes,
        classifier_activation='softmax'
    )

def get_model_bert():
    """
    https://www.tensorflow.org/text/tutorials/classify_text_with_bert#define_your_model
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    return tf.keras.Model(text_input, net)



def build_test(structure):
    if config.K_ALGO in structure:
        if structure[config.K_ALGO] == config.V_ALGO_INCEPTION:
            return InceptionTestRun(structure)
        elif structure[config.K_ALGO] == config.V_ALGO_BERT:
            return BertTestRun(structure)
        
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
    
    def get_batch_size(self):
        return int(self.structure[config.K_BATCH_SIZE])
    
    def get_pci_speed(self):
        return float(self.structure[config.K_PCI_PERC]) / 100.0
    
    def get_gpu_speed(self):
        return float(self.structure[config.K_GPU_PERC]) / 100.0

class InceptionTestRun(TestRun):
    pass
class BertTestRun(TestRun):
    pass