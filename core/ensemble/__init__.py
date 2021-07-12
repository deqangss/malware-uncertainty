from tools.utils import ensemble_method_scope as _ensemble_method_names
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import DeepEnsemble, WeightedDeepEnsemble
from core.ensemble.vanilla import Vanilla
from collections import namedtuple

_Ensemble_methods = namedtuple('ensemble_methods', _ensemble_method_names)
_ensemble_methods = _Ensemble_methods(vanilla=Vanilla,
                                      mc_dropout=MCDropout,
                                      bayesian=BayesianEnsemble,
                                      deep_ensemble=DeepEnsemble,
                                      weighted_ensemble=WeightedDeepEnsemble
                                      )
ensemble_method_scope_dict = dict(_ensemble_methods._asdict())
