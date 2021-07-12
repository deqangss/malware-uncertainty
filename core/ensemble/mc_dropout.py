from core.ensemble.vanilla import Vanilla, model_builder
from core.ensemble.model_hp import train_hparam, mc_dropout_hparam
from tools import utils
from config import logging, ErrorHandler

logger = logging.getLogger('ensemble.mc_dropout')
logger.addHandler(ErrorHandler)


class MCDropout(Vanilla):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=1,
                 model_directory=None,
                 name='MC_DROPOUT'
                 ):
        super(MCDropout, self).__init__(architecture_type,
                                        base_model,
                                        n_members,
                                        model_directory,
                                        name)
        self.hparam = utils.merge_namedtuples(train_hparam, mc_dropout_hparam)
        self.ensemble_type = 'mc_dropout'

    def build_model(self, input_dim=None):
        """
        Build an ensemble model -- only the homogeneous structure is considered
        :param input_dim: integer or list, input dimension shall be set in some cases under eager mode
        """
        callable_graph = model_builder(self.architecture_type)

        @callable_graph(input_dim, use_mc_dropout=True)
        def _builder():
            return utils.produce_layer(self.ensemble_type, dropout_rate=self.hparam.dropout_rate)

        self.base_model = _builder()
        return

    def model_generator(self):
        try:
            if len(self.weights_list) <= 0:
                self.load_ensemble_weights()
        except Exception as e:
            raise Exception("Cannot load model weights:{}.".format(str(e)))

        assert len(self.weights_list) == self.n_members
        self.base_model.set_weights(weights=self.weights_list[self.n_members - 1])
        # if len(self._optimizers_dict) > 0 and self.base_model.optimizer is not None:
        #     self.base_model.optimizer.set_weights(self._optimizers_dict[self.n_members - 1])
        for _ in range(self.hparam.n_sampling):
            yield self.base_model
