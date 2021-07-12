"""
base ensemble class
"""


class Ensemble(object):
    def __init__(self, architecture_type, base_model, n_members, model_directory):
        """
        initialization
        :param architecture_type: e.g., 'dnn'
        :param base_model: an instantiated object of base model
        :param n_members: number of base models
        :param model_directory: a folder for saving ensemble weights
        """
        self.architecture_type = architecture_type
        self.base_model = base_model
        self.n_members = n_members
        self.model_directory = model_directory
        self.weights_list = []  # a model's parameters
        self._optimizers_dict = dict()

    def build_model(self):
        """Build an ensemble model"""
        raise NotImplementedError

    def predict(self, x):
        """conduct prediction"""
        raise NotImplementedError

    def get_basic_layers(self):
        """ construct the basic layers"""
        raise NotImplementedError

    def fit(self, train_x, train_y, val_x=None, val_y=None, **kwargs):
        """ tune the model parameters upon given dataset"""
        raise NotImplementedError

    def get_model_number(self):
        """ get the number of base models"""
        raise NotImplementedError

    def reset(self):
        self.weights_list = []

    def save_ensemble_weights(self):
        """ save the model parameters"""
        raise NotImplementedError

    def load_ensemble_weights(self):
        """ Load the model parameters """
        raise NotImplementedError

    def gradient_loss_wrt_input(self, x):
        """ obtain gradients of loss function with respect to the input."""
        raise NotImplementedError
