# conduct the group of 'out of distribution' experiments on drebin dataset
import os
import sys
import random
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from core.feature import feature_type_scope_dict, feature_type_vs_architecture
from core.ensemble import ensemble_method_scope_dict
from tools import utils
from config import config, logging

logger = logging.getLogger('experiment.drebin_ood')

# procedure of ood experiments
# 1. build dataset
# 2. preprocess data
# 3. learn models
# 4. save results for statistical analysis

def run_experiment(feature_type, ensemble_type, n_members = 1, proc_numbers=2):
    """
    run this group of experiments
    :param feature_type: the type of features (e.g., drebin, opcode, etc.), feature type associates to the model architecture
    :param ensemble_type: the ensemble method (e.g., vanilla, deep_ensemble, etc.
    :return: None
    """
    mal_folder, ben_folder, ood_data_paths = build_data()

    train_dataset, validation_dataset, test_data, test_y, ood_data, ood_y, input_dim = \
        data_preprocessing(feature_type, mal_folder, ben_folder, ood_data_paths, proc_numbers)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation
    arch_type = feature_type_vs_architecture.get(feature_type)
    saving_dir = config.get('experiments', 'ood')
    if ensemble_type in ['vanilla', 'mc_dropout', 'bayesian']:
        ensemble_model = ensemble_obj(arch_type, base_model=None, n_members = 1, model_directory = saving_dir)
    else:
        ensemble_model = ensemble_obj(arch_type, base_model=None, n_members = n_members, model_directory = saving_dir)

    ensemble_model.fit(train_dataset, validation_dataset, input_dim=input_dim)

    test_results = ensemble_model.predict(test_data)
    utils.dump_joblib(test_results, os.path.join(saving_dir, '{}_{}_test.res'.format(feature_type, ensemble_type)))
    ensemble_model.evaluate(test_data, test_y)
    ood_results = ensemble_model.predict(ood_data)
    utils.dump_joblib(ood_results, os.path.join(saving_dir, '{}_{}_ood.res'.format(feature_type, ensemble_type)))
    ensemble_model.evaluate(ood_data, ood_y, is_single_class=True, name='ood')

def build_data():
    malware_dir = config.get('drebin', 'malware_dir')
    benware_dir = config.get('drebin', 'benware_dir')
    malare_paths, ood_data_paths = produce_ood_data(malware_dir)

    return malare_paths, benware_dir, ood_data_paths


def produce_ood_data(malware_dir, top_frequency=30, n_selection=5, minimum_samples = 1, maximum_samples=1000):
    import pandas as pd
    malware_family_pd = pd.read_csv(config.get('drebin', 'malware_family'))
    counter = dict(Counter(malware_family_pd['family']).most_common(top_frequency))
    i = 0
    while i <= 1e5:
        random.seed(i)
        selected_families = random.sample(counter.keys(), n_selection)
        number_of_malware = sum([counter[f] for f in selected_families])
        if minimum_samples <= number_of_malware <= maximum_samples:
            break
        else:
            i = i + 1
    else:
        random.seed(1)
        selected_families = random.sample(counter.keys(), n_selection)
        number_of_malware = sum([counter[f] for f in selected_families])
    logger.info('The number of selected ood malware samples: {}'.format(number_of_malware))
    logger.info("The selected families are {}".format(','.join(selected_families)))

    selected_sha256_codes = list(malware_family_pd[malware_family_pd.family.isin(selected_families)]['sha256'])
    assert len(selected_sha256_codes) == number_of_malware

    malware_paths = utils.retrive_files_set(malware_dir, "", ".apk|")
    ood_data_paths = []
    for mal_path in malware_paths:
        sha256 = utils.get_sha256(mal_path)
        if sha256 in selected_sha256_codes:
            ood_data_paths.append(mal_path)

    return list(set(malware_paths) - set(ood_data_paths)), ood_data_paths


def data_preprocessing(feature_type='drebin', malware_dir=None, benware_dir=None, ood_data_path = None, proc_numbers = 2):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())

    android_features_saving_dir = config.get('metadata', 'naive_data_directory')
    intermediate_data_saving_dir = config.get('metadata', 'meta_data_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number = proc_numbers)
    mal_feature_list = feature_extractor.feature_extraction(malware_dir)
    n_malware = len(mal_feature_list)
    ben_feature_list = feature_extractor.feature_extraction(benware_dir)
    n_benware = len(ben_feature_list)
    feature_list = mal_feature_list + ben_feature_list
    gt_labels = np.zeros((len(feature_list),), dtype=np.int32)
    gt_labels[:n_malware] = 1

    # data split
    train_features, test_features, train_y, test_y = train_test_split(feature_list, gt_labels,
                                                                      test_size=0.2, random_state=0)
    feature_extractor.feature_preprocess(train_features, train_y)  # produce intermediate products
    # obtain validation data
    train_features, validation_features, train_y, validation_y = train_test_split(train_features,
                                                                                  train_y,
                                                                                  test_size=0.25,
                                                                                  random_state=0
                                                                                  )

    # obtain data in a format for ML algorithms
    train_dataset, input_dim = feature_extractor.feature2ipt(train_features, train_y)
    test_data, _ = feature_extractor.feature2ipt(test_features)
    validation_dataset, _ = feature_extractor.feature2ipt(validation_features, validation_y)

    ood_features = feature_extractor.feature_extraction(ood_data_path)
    ood_y = np.ones((len(ood_features),))
    ood_data, _ = feature_extractor.feature2ipt(ood_features)

    return train_dataset, validation_dataset, test_data, test_y, ood_data, ood_y, input_dim


def get_ensemble_object(ensemble_type):
    assert ensemble_type in ensemble_method_scope_dict.keys(), '{} expected, but {} are supported'.format(
        ensemble_type,
        ','.join(ensemble_method_scope_dict.keys())
    )
    return ensemble_method_scope_dict[ensemble_type]

def _main():
    # build_data()
    print(get_ensemble_object('vanilla'))


if __name__ == '__main__':
    sys.exit(_main())
