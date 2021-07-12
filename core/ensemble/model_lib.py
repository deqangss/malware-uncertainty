""" This script is for building model graph"""

import tensorflow as tf

from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('core.ensemble.model_lib')
logger.addHandler(ErrorHandler)


def model_builder(architecture_type='dnn'):
    assert architecture_type in model_name_type_dict, 'models are {}'.format(','.join(model_name_type_dict.keys()))
    return model_name_type_dict[architecture_type]


def _change_scaler_to_list(scaler):
    if not isinstance(scaler, (list, tuple)):
        return [scaler]
    else:
        return scaler


def _dnn_graph(input_dim=None, use_mc_dropout=False):
    """
    The deep neural network based malware detector.
    The implement is based on the paper, entitled ``Adversarial Examples for Malware Detection'',
    which can be found here:  http://patrickmcdaniel.org/pubs/esorics17.pdf

    We slightly change the model architecture by reducing the number of neurons at the last layer to one.
    """
    input_dim = _change_scaler_to_list(input_dim)
    from core.ensemble.model_hp import dnn_hparam
    logger.info(dict(dnn_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, _1, _2, _3 = func()
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.InputLayer(input_shape=(input_dim[0],)))
            for units in dnn_hparam.hidden_units:
                model.add(Dense(units, activation=dnn_hparam.activation))
            if use_mc_dropout:
                model.add(tf.keras.layers.Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            else:
                model.add(tf.keras.layers.Dropout(dnn_hparam.dropout_rate))
                model.add(Dense(dnn_hparam.output_dim, activation=tf.nn.sigmoid))
            return model

        return graph

    return wrapper


def _text_cnn_graph(input_dim=None, use_mc_dropout=False):
    """
    deep android malware detection
    The implement is based on the paper, entitled ``Deep Android Malware Detection'',
    which can be found here:  https://dl.acm.org/doi/10.1145/3029806.3029823
    """
    input_dim = _change_scaler_to_list(input_dim)  # dynamical input shape is permitted
    from core.ensemble.model_hp import text_cnn_hparam
    logger.info(dict(text_cnn_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, Conv2D, _1, _2 = func()

            class TextCNN(tf.keras.models.Model):
                def __init__(self):
                    super(TextCNN, self).__init__()
                    self.embedding = tf.keras.layers.Embedding(text_cnn_hparam.vocab_size,
                                                               text_cnn_hparam.n_embedding_dim)
                    self.spatial_dropout = tf.keras.layers.SpatialDropout2D(rate=text_cnn_hparam.dropout_rate)
                    self.conv = Conv2D(text_cnn_hparam.n_conv_filters, text_cnn_hparam.kernel_size,
                                       activation=text_cnn_hparam.activation)
                    self.conv_dropout = tf.keras.layers.Dropout(rate=text_cnn_hparam.dropout_rate)
                    self.pooling = tf.keras.layers.GlobalMaxPool2D()  # produce a fixed length vector
                    self.denses = [Dense(neurons, activation='relu') for neurons in text_cnn_hparam.hidden_units]
                    self.dropout = tf.keras.layers.Dropout(text_cnn_hparam.dropout_rate)
                    if use_mc_dropout:
                        self.d_out = tf.keras.layers.Dense(text_cnn_hparam.output_dim, activation=tf.nn.sigmoid)
                    else:
                        self.d_out = Dense(text_cnn_hparam.output_dim, activation=tf.nn.sigmoid)

                def call(self, x, training=False):
                    embed_code = self.embedding(x)
                    # batch_size, seq_length, embedding_dim, 1. Note: seq_length >= conv_kernel_size
                    embed_code = tf.expand_dims(embed_code, axis=-1)
                    if text_cnn_hparam.use_spatial_dropout:
                        embed_code = self.spatial_dropout(embed_code, training=training)

                    conv_x = self.conv(embed_code)
                    if text_cnn_hparam.use_conv_dropout:
                        conv_x = self.conv_dropout(conv_x)

                    flatten_x = self.pooling(conv_x)

                    for i, dense in enumerate(self.denses):
                        flatten_x = dense(flatten_x)
                    if not use_mc_dropout:
                        flatten_x = self.dropout(flatten_x, training=training)
                    return self.d_out(flatten_x)

            return TextCNN()

        return graph

    return wrapper


def _multimodalitynn(input_dim=None, use_mc_dropout=False):
    """
    A Multimodal Deep Learning Method for Android Malware Detection Using Various Features

    The implement is based on our understanding of the paper, entitled
    ``A Multimodal Deep Learning Method for Android Malware Detection Using Various Features'':
    @ARTICLE{8443370,
      author={T. {Kim} and B. {Kang} and M. {Rho} and S. {Sezer} and E. G. {Im}},
      journal={IEEE Transactions on Information Forensics and Security},
      title={A Multimodal Deep Learning Method for Android Malware Detection Using Various Features},
      year={2019},
      volume={14},
      number={3},
      pages={773-788},}
    """
    input_dim = _change_scaler_to_list(input_dim)
    assert isinstance(input_dim, (list, tuple)), 'a list of input dimensions are mandatory.'
    from core.ensemble.model_hp import multimodalitynn_hparam
    assert len(input_dim) == multimodalitynn_hparam.n_modalities, 'Expected input number {}, but got {}'.format(
        multimodalitynn_hparam.n_modalities, len(input_dim))
    logger.info(dict(multimodalitynn_hparam._asdict()))

    def wrapper(func):
        def graph():
            input_layers = []
            Dense, _1, _2, _3 = func()

            for idx, header in enumerate(range(multimodalitynn_hparam.n_modalities)):
                input_layers.append(
                    tf.keras.Input(input_dim[idx], name='HEADER_{}'.format(idx + 1))
                )

            x_initial_out = []
            for x in input_layers:
                for units in multimodalitynn_hparam.initial_hidden_units:
                    x = Dense(units, activation=multimodalitynn_hparam.activation)(x)
                x_initial_out.append(x)
            x_out = tf.keras.layers.concatenate(x_initial_out)
            for units in multimodalitynn_hparam.hidden_units:
                x_out = Dense(units, activation=multimodalitynn_hparam.activation)(x_out)

            if use_mc_dropout:
                out = tf.keras.layers.Dense(multimodalitynn_hparam.output_dim, activation=tf.nn.sigmoid)(x_out)
            else:
                out = tf.keras.layers.Dropout(rate=multimodalitynn_hparam.dropout_rate)(x_out)
                out = Dense(multimodalitynn_hparam.output_dim, activation=tf.nn.sigmoid)(out)
            return tf.keras.Model(inputs=input_layers, outputs=out)

        return graph

    return wrapper


def _r2d2(input_dim=None, use_mc_dropout=False):
    """
    R2-D2: ColoR-inspired Convolutional NeuRal Network (CNN)-based AndroiD Malware Detections

    The implement is based on our understanding of the paper, entitled
    ``R2-D2: ColoR-inspired Convolutional NeuRal Network (CNN)-based AndroiD Malware Detections'':
    @INPROCEEDINGS{8622324,
      author={T. H. {Huang} and H. {Kao}},
      booktitle={2018 IEEE International Conference on Big Data (Big Data)},
      title={R2-D2: ColoR-inspired Convolutional NeuRal Network (CNN)-based AndroiD Malware Detections},
      year={2018},
      volume={},
      number={},
      pages={2633-2642},}
    """
    input_dim = _change_scaler_to_list(input_dim)
    from core.ensemble.model_hp import r2d2_hparam
    logger.info(dict(r2d2_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, _1, _2, last_Dense = func()
            base_model = tf.keras.applications.MobileNetV2(input_shape=input_dim,
                                                           include_top=False,
                                                           weights='imagenet')
            base_model.trainable = False
            for layer in base_model.layers[-r2d2_hparam.unfreezed_layers:]:
                layer.trainable = True
            x_new = base_model.layers[-1].output

            x_new = tf.keras.layers.GlobalAveragePooling2D()(x_new)
            if use_mc_dropout:
                # x_new = tf.nn.dropout(x_new, rate=mc_droput_rate)
                # out = tf.keras.layers.Dense(r2d2_hparam.output_dim, activation=tf.nn.sigmoid)(x_new)
                out = last_Dense(r2d2_hparam.output_dim, activation=tf.nn.sigmoid)(x_new)
            else:
                x_new = tf.keras.layers.Dropout(r2d2_hparam.dropout_rate)(x_new)
                out = Dense(r2d2_hparam.output_dim, activation=tf.nn.sigmoid)(x_new)
            return tf.keras.Model(inputs=base_model.input, outputs=out)

        return graph

    return wrapper


def _droidectc_graph(input_dim=None, use_mc_dropout=False):
    """
    DROIDETEC: Android Malware Detection and Malicious Code Localization through Deep Learning

    The implement is based on our understanding of the paper, entitled
    ``DROIDETEC: Android Malware Detection and Malicious Code Localization through Deep Learning'':
    @article{ma2020droidetec,
      title={Droidetec: Android malware detection and malicious code localization through deep learning},
      author={Ma, Zhuo and Ge, Haoran and Wang, Zhuzhu and Liu, Yang and Liu, Ximeng},
      journal={arXiv preprint arXiv:2002.03594},
      year={2020}
    }
    """
    input_dim = _change_scaler_to_list(input_dim)  # dynamic input shape is permitted
    from core.ensemble.model_hp import droidetec_hparam
    logger.info(dict(droidetec_hparam._asdict()))

    def wrapper(func):
        def graph():
            Dense, _1, LSTM, last_Dense = func()

            class BiLSTMAttention(tf.keras.models.Model):
                def __init__(self):
                    super(BiLSTMAttention, self).__init__()
                    self.embedding = tf.keras.layers.Embedding(droidetec_hparam.vocab_size,
                                                               droidetec_hparam.n_embedding_dim)

                    self.bi_lstm = tf.keras.layers.Bidirectional(LSTM(droidetec_hparam.lstm_units,
                                                                      return_sequences=True),
                                                                 merge_mode='sum'
                                                                 )
                    self.dense_layer = tf.keras.layers.Dense(droidetec_hparam.lstm_units, use_bias=False)
                    # for units in droidetec_hparam.hidden_units:
                    #     self.dense_layers.append(Dense(droidetec_hparam.hidden_units, use_bias=False))
                    if use_mc_dropout:
                        self.output_layer = last_Dense(droidetec_hparam.output_dim, activation=tf.nn.sigmoid)
                    else:
                        self.output_layer = Dense(droidetec_hparam.output_dim, activation=tf.nn.sigmoid)

                def call(self, x, training=False):
                    embed_x = self.embedding(x)
                    # if use_mc_dropout:
                    #     stateful_x = self.bi_lstm(embed_x, training=True)
                    # else:
                    stateful_x = self.bi_lstm(embed_x)
                    alpha_wights = tf.nn.softmax(self.dense_layer(tf.nn.tanh(stateful_x)), axis=1)
                    attn_x = tf.reduce_sum(alpha_wights * stateful_x, axis=1)
                    # if use_mc_dropout:
                    #     attn_x = tf.nn.dropout(attn_x, rate=mc_dropout_rate)
                    return self.output_layer(attn_x)

            return BiLSTMAttention()

        return graph

    return wrapper


model_name_type_dict = {
    'dnn': _dnn_graph,
    'text_cnn': _text_cnn_graph,
    'multimodalitynn': _multimodalitynn,
    'r2d2': _r2d2,
    'droidectc': _droidectc_graph
}


def build_models(input_x, architecture_type, ensemble_type='vanilla', input_dim=None, use_mc_dropout=False):
    builder = model_builder(architecture_type)

    @builder(input_dim, use_mc_dropout)
    def graph():
        return utils.produce_layer(ensemble_type, dropout_rate=0.4)

    model = graph()
    return model(input_x)
