import tensorflow as tf
import os

from keras import backend
from keras.models import load_model


def keras2tf(keras_hdf5, tfckpt, tf_graph):
    backend.set_learning_phase(0)

    loaded_model = load_model(keras_hdf5)

    print('Keras model information:')
    print(' Input names :', loaded_model.inputs)
    print(' Output names:', loaded_model.outputs)
    print('-------------------------------------')

    saver = tf.train.Saver()

    tf_session = backend.get_session()

    input_graph_def = tf_session.graph.as_graph_def()

    tfgraph_path = './train'
    tfgraph_filename = 'tf_complete_model.pb'

    saver.save(tf_session, tfckpt)
    tf.train.write_graph(input_graph_def, tfgraph_path, tfgraph_filename, as_text=False)

    print('TensorFlow information:')
    print(' Checkpoint saved as:', tfckpt)
    print(' Graph saved as     :', os.path.join(tfgraph_path, tfgraph_filename))
    print('-------------------------------------')
    return
