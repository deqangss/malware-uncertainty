import os
import time
import warnings

from collections import defaultdict
from androguard.misc import AnalyzeAPK
from tools import utils
from config import logging

current_dir = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger("feature.apiseq")

REMOVE_CLASS_HEAD_LIST = [
    'Ljava/', 'Ljavax/'
]

RETAIN_CLASS_HEAD_LIST = [
    'Landroid/',
    'Lcom/android/internal/util/',
    'Ldalvik/',
    'Lorg/apache/',
    'Lorg/json/',
    'Lorg/w3c/dom/',
    'Lorg/xml/sax',
    'Lorg/xmlpull/v1/',
    'Ljunit/'
]


def _check_class(class_name):
    for cls_head in REMOVE_CLASS_HEAD_LIST:
        if class_name.startswith(cls_head):
            return False
    for cls_head in RETAIN_CLASS_HEAD_LIST:
        if class_name.startswith(cls_head):
            return True

    return False


def _dfs(api, nodes, seq=[], visited=[]):
    if api not in nodes.keys():
        seq.append(api)
    else:
        visited.append(api)
        for elem in nodes[api]:
            if elem in visited:
                seq.append(elem)
            else:
                _dfs(elem, nodes, seq, visited)


def get_api_sequence(apk_path, save_path):
    """
    produce an api call sequence for an apk
    :param apk_path: an apk path
    :param save_path: path for saving resulting feature
    :return: (status, back_path_name)
    """
    try:
        # obtain tow dictionaries: xref_from and xref_to, of which key is class-method name
        # and value is the caller or callee.
        _, _, dx = AnalyzeAPK(apk_path)
        mth_callers = defaultdict(list)
        mth_callees = defaultdict(list)
        for cls_obj in dx.get_classes():  # ClassAnalysis
            if cls_obj.is_external():
                continue
            cls_name = cls_obj.name
            for mth_obj in cls_obj.get_methods():
                if mth_obj.is_external():
                    continue

                m = mth_obj.get_method()  # dvm.EncodedMethod
                cls_mth_name = cls_name + '->' + m.name + m.proto
                # get callers
                mth_callers[cls_mth_name] = []
                for _, call, _ in mth_obj.get_xref_from():
                    if _check_class(call.class_name):
                        mth_callers[cls_mth_name].append(call.class_name + '->' + call.name + call.proto)
                # get callees sequentially
                for instruction in m.get_instructions():
                    opcode = instruction.get_name()
                    if 'invoke-' in opcode:
                        code_body = instruction.get_output()
                        if '->' not in code_body:
                            continue
                        head_part, rear_part = code_body.split('->')
                        class_name = head_part.strip().split(' ')[-1]
                        mth_name_callee = class_name + '->' + rear_part
                        if _check_class(mth_name_callee):
                            mth_callees[cls_mth_name].append(mth_name_callee)

        # look for the root call
        root_calls = []
        num_of_calls = len(mth_callers.items())
        if num_of_calls == 0:
            raise ValueError("No callers")
        for k in mth_callers.keys():
            if (len(mth_callers[k]) <= 0) and (len(mth_callees[k]) > 0):
                root_calls.append(k)

        if len(root_calls) == 0:
            warnings.warn("Cannot find a root call, instead, randomly pick up one.")
            import random
            id = random.choice(range(num_of_calls))
            root_calls.append(mth_callers.keys()[id])

        # generate sequence
        api_sequence = []
        for root_call in root_calls:
            sub_seq = []
            visited_nodes = []
            _dfs(root_call, mth_callees, sub_seq, visited_nodes)
            api_sequence.extend(sub_seq)
        # dump feature
        utils.dump_txt('\n'.join(api_sequence), save_path)
        return save_path
    except Exception as e:
        if len(e.args) > 0:
            e.args = e.args + (apk_path,)
        return e


def load_feature(save_path):
    return utils.read_txt(save_path)


def wrapper_load_feature(save_path):
    try:
        return load_feature(save_path)
    except Exception as e:
        return e


def _mapping(path_to_feature, dictionary):
    features = load_feature(path_to_feature)
    _feature = [idx for idx in list(map(dictionary.get, features)) if idx is not None]
    if len(_feature) == 0:
        _feature = [dictionary.get('sos')]
        warnings.warn("Produce zero feature vector.")
    return _feature


def wrapper_mapping(ptuple):
    try:
        return _mapping(*ptuple)
    except Exception as e:
        return e
