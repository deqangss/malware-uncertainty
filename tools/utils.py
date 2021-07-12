from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

import os
import sys
import shutil
import hashlib
import warnings
import functools
from collections import namedtuple, defaultdict


class ParamWrapper(object):
    def __init__(self, params):
        if not isinstance(params, dict):
            params = vars(params)
        self.params = params

    def __getattr__(self, name):
        val = self.params.get(name)
        if val is None:
            MSG = "Setting params ({}) is deprecated"
            warnings.warn(MSG.format(name))
        return val


def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """

    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir, topdown=True):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path, file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_list = list()
    for ext in file_exts:
        file_path_list.extend(get_file_name(os.path.join(base_dir, dir_ext), ext))

    # remove duplicate elements
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(file_path_list))


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_file_nameext(path):
    return os.path.basename(path)


def dump_pickle(data, path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True


def read_pickle(path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def read_txt(path, mode='r'):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
            return lines
    else:
        raise ValueError("{} does not seen like a file path.\n".format(path))


def dump_txt(data_str, path, mode='w'):
    if not isinstance(data_str, str):
        raise TypeError

    with open(path, mode) as f_w:
        f_w.write(data_str)


def readdata_np(data_path):
    try:
        with open(data_path, 'rb') as f_r:
            data = np.load(f_r)
        return data
    except IOError as e:
        raise IOError("Unable to open {0}: {1}.\n".format(data_path, str(e)))


def dumpdata_np(data, data_path):
    if not isinstance(data, np.ndarray):
        warnings.warn("The array is not the numpy.ndarray type.")
    data_dir = os.path.dirname(data_path)
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(data_path, 'wb') as f_s:
            np.save(f_s, data)
    except OSError as e:
        sys.stderr.write(e)


def safe_load_json(json_path):
    try:
        import yaml
        with open(json_path, 'r') as rh:
            return yaml.safe_load(rh)
    except IOError as ex:
        raise IOError(str(ex) + ": Unable to load json file.")


def load_json(json_path):
    try:
        import json
        with open(json_path, 'r') as rh:
            return json.load(rh)
    except IOError as ex:
        raise IOError(str(ex) + ": Unable to load json file.")


def dump_json(obj_dict, file_path):
    try:
        import json
        if not os.path.exists(os.path.dirname(file_path)):
            mkdir(os.path.dirname(file_path))

        with open(file_path, 'w') as fh:
            json.dump(obj_dict, fh)
    except IOError as ex:
        raise IOError(str(ex) + ": Fail to dump dict using json toolbox")


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def copy_files(src_file_list, dst_dir):
    if not isinstance(src_file_list, list):
        raise TypeError
    if os.path.isdir(dst_dir):
        raise ValueError
    for src in src_file_list:
        if not os.path.isfile(src):
            continue
        shutil.copy(src, dst_dir)


def get_sha256(file_path):
    assert os.path.isfile(file_path), 'permit only file path'
    fh = open(file_path, 'rb')
    sha256 = hashlib.sha256()
    while True:
        data = fh.read(8192)
        if not data:
            break
        sha256.update(data)
    fh.close()
    return sha256.hexdigest()


def merge_namedtuples(tp1, tp2):
    from collections import namedtuple
    _TP12 = namedtuple('tp12', tp1._fields + tp2._fields)
    return _TP12(*(tp1 + tp2))


def expformat(f, pos, prec=0, exp_digits=1, sign='off'):
    """Scientific-format a number with a given number of digits in the exponent.
    Optionally remove the sign in the exponent"""
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    if sign == 'on':
        # add 1 to digits as 1 is taken by sign +/-
        return "%se%+0*d" % (mantissa, exp_digits+1, int(exp))
    else :
        return "%se%0*d" % (mantissa, exp_digits, int(exp))


def bootstrap(data, fun, n_resamples=1000, alpha=0.05, seed=0):
    """Compute confidence interval for values of function fun

    Parameters
    ==========
    data: list of arguments to fun

    """
    assert isinstance(data, list)
    n_samples = len(data[0])
    np.random.seed(seed)
    idx = np.random.randint(0, n_samples, (n_resamples, n_samples))

    def select(sample):
        return [d[sample] for d in data]

    def evaluate(sample):
        result = select(sample)
        values = []
        for elems in zip(*result):
            values.append(fun(*elems))
        return np.stack(values, axis=0)

    values = evaluate(idx)

    idx = idx[np.argsort(values, axis=0, kind='mergesort')]
    values = np.sort(values, axis=0, kind='mergesort')

    stat = namedtuple('stat', ['value', 'index'])
    low = stat(value=values[int((alpha / 2.0) * n_resamples)],
               index=idx[int((alpha / 2.0) * n_resamples)])
    high = stat(value=values[int((1 - alpha / 2.0) * n_resamples)],
                index=idx[int((1 - alpha / 2.0) * n_resamples)])
    mean = stat(value=np.mean(values, axis=0),
                index=None)

    return low, high, mean


########################################################################################
############################# functions for tf models ##################################
########################################################################################
ensemble_method_scope = ['vanilla', 'mc_dropout', 'deep_ensemble', 'weighted_ensemble', 'bayesian']


class DenseDropout(tf.keras.layers.Layer):
    def __init__(self, units,
                 dropout_rate,
                 activation=None,
                 use_dropout=True,
                 **kwargs):
        """
        Initialize a dense-dropout layer
        :param units: number of neurons
        :param dropout_rate: a float value between 0 and 1. A portion of activations will be dropped randomly
        :param activation: activation function
        param use_dropout: performing dropout in both training and testing phases
        :param kwargs: other arguments for tf.keras.layers.Dense
        """
        super(DenseDropout, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.kwargs = kwargs
        self.dense_layer = tf.keras.layers.Dense(units, activation=self.activation, **self.kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=True):
        return self.dropout_layer(self.dense_layer(inputs), training=self.use_dropout)


class Conv2DDropout(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dropout_rate,
                 activation=None,
                 use_dropout=True,
                 **kwargs):
        """
        Initialize a convolution-dropout layer
        :param filters: Positive integer, number of ouput channels
        :param kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of 2D convolution window
        :param dropout_rate: a float value between 0 and 1. A portion of activations will be dropped randomly
        :param activation: activation function
        :param use_dropout: performing dropout in both training and testing phases
        :param kwargs: other arguments for tf.keras.layers.Conv2D
        """
        super(Conv2DDropout, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.conv2d_layer = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, **kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=True):
        return self.dropout_layer(self.conv2d_layer(inputs), training=self.use_dropout)


class LSTMDropout(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 dropout_rate,
                 use_dropout=True,
                 go_backwards=True,
                 return_sequences=True, **kwargs):
        """
        Initialize a LSTM-dropout layer
        :param dropout_rate: a float value between 0 and 1. A portion of activations will be dropped randomly
        :param units: Positive Integer, number of neurons
        :param use_dropout: performing dropout in both training and testing phases
        :param kwargs: other arguments for tf.keras.layers.LSTM
        """
        super(LSTMDropout, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.lstm = tf.keras.layers.LSTM(units, dropout=self.dropout_rate, return_sequences=self.return_sequences,
                                         **kwargs)

    def call(self, inputs, training=True):
        return self.lstm(inputs, training=self.use_dropout)

    @property
    def return_state(self):
        return self.lstm.return_state

    def get_config(self):
        config = super(LSTMDropout, self).get_config()
        config['dropout_rate'] = self.dropout_rate
        config['units'] = self.units
        config['use_dropout'] = self.use_dropout
        config['go_backwards'] = self.go_backwards
        return config


class DropoutDense(tf.keras.layers.Layer):
    def __init__(self, units,
                 dropout_rate,
                 activation=None,
                 use_dropout=True,
                 **kwargs):
        """
        Initialize a dense-dropout layer
        :param units: number of neurons
        :param dropout_rate: a float value between 0 and 1. A portion of activations will be dropped randomly
        :param activation: activation function
        param use_dropout: performing dropout in both training and testing phases
        :param kwargs: other arguments for tf.keras.layers.Dense
        """
        super(DropoutDense, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.kwargs = kwargs
        self.dense_layer = tf.keras.layers.Dense(units, activation=self.activation, **self.kwargs)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=True):
        return self.dense_layer(self.dropout_layer(inputs, training=self.use_dropout))


def dense_dropout(dropout_rate=0.4):
    return functools.partial(DenseDropout, dropout_rate=dropout_rate)


def conv2d_dropout(dropout_rate=0.4):
    return functools.partial(Conv2DDropout, dropout_rate=dropout_rate)


def lstm_dropout(dropout_rate=0.4):
    return functools.partial(LSTMDropout, dropout_rate=dropout_rate)


def dropout_dense(dropout_rate=0.4):
    return functools.partial(DropoutDense, dropout_rate=dropout_rate)


def scaled_reparameterization_layer(tfp_varitional_layer_obj, scale_factor=1. / 10000):
    def scaled_kl_fn(q, p, _):
        return tfp.distributions.kl_divergence(q, p) * scale_factor

    return functools.partial(tfp_varitional_layer_obj,
                             kernel_divergence_fn=scaled_kl_fn,
                             bias_divergence_fn=scaled_kl_fn)


def customized_reparameterization_dense_layer(scale_factor=1. / 10000):
    # code from: https://github.com/tensorflow/probability/issues/409
    # and https://github.com/tensorflow/probability/blob/v0.11.0/tensorflow_probability/python/layers/util.py#L202-L224
    tfd = tfp.distributions

    def _posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n],
                           scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    def _non_trainable_prior_fn(kernel_size, bias_size=0, dtype=None):
        def _distribution_fn(_):
            return tfd.Independent(tfd.Normal(loc=tf.zeros(kernel_size + bias_size, dtype=dtype),
                                              scale=1.),
                                   reinterpreted_batch_ndims=1)

        return _distribution_fn

    def _trainable_prior_fn(kernel_size, bias_size=0, dtype=None):
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(kernel_size + bias_size, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda mu: tfd.Independent(tfd.Normal(loc=mu, scale=1),
                                           reinterpreted_batch_ndims=1)),
        ])

    return functools.partial(
        tfp.layers.DenseVariational,
        make_posterior_fn=_posterior_mean_field,
        make_prior_fn=_non_trainable_prior_fn,
        kl_weight=scale_factor)


def produce_layer(ensemble_type=None, **kwargs):
    assert ensemble_type in ensemble_method_scope, 'only support ensemble method {}.'.format(
        ','.join(ensemble_method_scope)
    )
    if ensemble_type == 'vanilla' or ensemble_type == 'deep_ensemble' or ensemble_type == 'weighted_ensemble':
        Dense = tf.keras.layers.Dense
        Conv2D = tf.keras.layers.Conv2D
        LSTM = tf.keras.layers.LSTM
        last_Dense = tf.keras.layers.Dense
    elif ensemble_type == 'mc_dropout':
        Dense = dense_dropout(kwargs['dropout_rate'])
        Conv2D = conv2d_dropout(kwargs['dropout_rate'])
        LSTM = lstm_dropout(kwargs['dropout_rate'])
        last_Dense = dropout_dense(kwargs['dropout_rate'])
    elif ensemble_type == 'bayesian':
        Dense = scaled_reparameterization_layer(tfp.layers.DenseReparameterization, kwargs['kl_scaler']) # customized_reparameterization_dense_layer(kwargs['kl_scaler']) #
        Conv2D = scaled_reparameterization_layer(tfp.layers.Convolution2DReparameterization, kwargs['kl_scaler'])
        LSTM = tf.keras.layers.LSTM
        last_Dense = scaled_reparameterization_layer(tfp.layers.DenseReparameterization, kwargs['kl_scaler']) # customized_reparameterization_dense_layer(kwargs['kl_scaler']) #
    else:
        raise ValueError('only support ensemble method {}.'.format(','.join(ensemble_method_scope)))

    return Dense, Conv2D, LSTM, last_Dense


##### neural network initialization ###########

def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def glorot_uniform(shape):
    if len(shape) > 1:
        fan_in, fan_out = get_fans(shape)
        scale = np.sqrt(6. / (fan_in + fan_out))
        return np.random.uniform(low=-scale, high=scale, size=shape)
    else:
        return np.zeros(shape, dtype=np.float32)
