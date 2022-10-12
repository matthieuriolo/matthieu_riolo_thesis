#!/usr/bin/env python3


def reset_weights(model):
    """
    resets the weights of a model
    Special thanks to https://github.com/keras-team/keras/issues/341
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))
            
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))

            var.assign(initializer(var.shape, var.dtype)) # use the initializer




# build inception from tensorflow application

tf.keras.applications.inception_v3.InceptionV3(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)


# https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1

model = tfhub.load('https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1')


# import tarfile
# file = tarfile.open('bert.tar.gz')
# file.extractall('./tmp')  
# file.close()

# from pathlib import Path
# filename = Path('bert.tar.gz')
# response = requests.get('https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1?tf-hub-format=compressed')
# filename.write_bytes(response.content)