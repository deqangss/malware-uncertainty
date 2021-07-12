import os
import time
import warnings

from pprint import pprint
import re
import collections
from androguard.misc import APK, AnalyzeAPK
import lxml.etree as etree
from xml.dom import minidom

from tools import utils
from sys import platform as _platform
from config import logging

if _platform == "linux" or _platform == "linux2":
    TMP_DIR = '/tmp/'
elif _platform == "win32" or _platform == "win64":
    TMP_DIR = 'C:\\TEMP\\'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

logger = logging.getLogger('feature.drebin')
current_dir = os.path.dirname(os.path.realpath(__file__))

# information about feature extraction
SuspiciousNames = ["getExternalStorageDirectory",
                   "getSimCountryIso",
                   "execHttpRequest",
                   "sendTextMessage",
                   "getPackageInfo",
                   "getSystemService",
                   "setWifiDisabled",
                   "Cipher",
                   "Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)",
                   "Ljava/io/IOException;->printStackTrace",
                   "Ljava/lang/Runtime;->exec",
                   "system/bin/su"
                   ]


def get_drebin_feature(apk_path, pmap, save_path):
    """
    produce `drebin' feature (dumped as a .data file) for a give path
    :param apk_path: an absolute path of an apk file
    :param pmap: api mapping class
    :param save_path: a folder (absolute path) for saving .data files
    :return: True or False,  `True' means conducting feature extraction successfully
    """
    try:
        print("Processing " + apk_path)
        start_time = time.time()
        data_dict = {}

        requested_permission_list, \
        activity_list, \
        service_list, \
        content_provider_list, \
        broadcast_receiver_list, \
        hardware_list, \
        intentfilter_list = get_feature_xml(apk_path)

        used_permission_list, \
        restricted_api_list, \
        suspicious_api_list, \
        url_list = get_feature_dex(apk_path, pmap, requested_permission_list)
        data_dict['requested_permission_list'] = requested_permission_list
        data_dict['activity_list'] = activity_list
        data_dict['service_list'] = service_list
        data_dict['content_provider_list'] = content_provider_list
        data_dict['broadcast_receiver_list'] = broadcast_receiver_list
        data_dict['hardware_list'] = hardware_list
        data_dict['broadcast_receiver_list'] = broadcast_receiver_list
        data_dict['hardware_list'] = hardware_list
        data_dict['intentfilter_list'] = intentfilter_list
        data_dict['used_permission_list'] = used_permission_list
        data_dict['restricted_api_list'] = restricted_api_list
        data_dict['suspicious_api_list'] = suspicious_api_list
        data_dict['url_list'] = url_list
        # feature = sum(
        #     [list(map(lambda elem_str: head_str + elem_str, body_list)) \
        #      for head_str, body_list in data_dict.items()],
        #     [])

        # dump the feature
        dump_feature(save_path, data_dict)
        return save_path
    except Exception as e:
        e.args += (apk_path,)
        return e


def get_feature_xml(apk_path):
    """
    get requested feature from manifest file
    :param apk_path: absolute path of an apk file
    :return: tuple of lists
    """
    requested_permission_list = []
    activity_list = []
    service_list = []
    content_provider_list = []
    broadcast_receiver_list = []
    hardware_list = []
    intentfilter_list = []
    xml_tmp_dir = os.path.join(TMP_DIR, 'xml_dir')
    if not os.path.exists(xml_tmp_dir):
        os.mkdir(xml_tmp_dir)
    apk_name = os.path.splitext(os.path.basename(apk_path))[0]
    try:
        apk_path = os.path.abspath(apk_path)
        a = APK(apk_path)
        f = open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'wb')
        xmlstreaming = etree.tostring(a.xml['AndroidManifest.xml'], pretty_print=True, encoding='utf-8')
        f.write(xmlstreaming)
        f.close()
    except Exception as e:
        raise Exception("Fail to load xml file of apk {}:{}".format(apk_path) + str(e))

    # start obtain feature S1, S2, S3, S4
    try:
        with open(os.path.join(xml_tmp_dir, apk_name + '.xml'), 'rb') as f:
            dom_xml = minidom.parse(f)
            dom_elements = dom_xml.documentElement

            dom_permissions = dom_elements.getElementsByTagName('uses-permission')
            for permission in dom_permissions:
                if permission.hasAttribute('android:name'):
                    requested_permission_list.append(permission.getAttribute('android:name'))

            dom_activities = dom_elements.getElementsByTagName('activity')
            for activity in dom_activities:
                if activity.hasAttribute('android:name'):
                    activity_list.append(activity.getAttribute('android:name'))

            dom_services = dom_elements.getElementsByTagName("service")
            for service in dom_services:
                if service.hasAttribute("android:name"):
                    service_list.append(service.getAttribute("android:name"))

            dom_contentproviders = dom_elements.getElementsByTagName("provider")
            for provider in dom_contentproviders:
                if provider.hasAttribute("android:name"):
                    content_provider_list.append(provider.getAttribute("android:name"))

            dom_broadcastreceivers = dom_elements.getElementsByTagName("receiver")
            for receiver in dom_broadcastreceivers:
                if receiver.hasAttribute("android:name"):
                    broadcast_receiver_list.append(receiver.getAttribute("android:name"))

            dom_hardwares = dom_elements.getElementsByTagName("uses-feature")
            for hardware in dom_hardwares:
                if hardware.hasAttribute("android:name"):
                    hardware_list.append(hardware.getAttribute("android:name"))

            dom_intentfilter_actions = dom_elements.getElementsByTagName("action")
            for action in dom_intentfilter_actions:
                if action.hasAttribute("android:name"):
                    intentfilter_list.append(action.getAttribute("android:name"))

            return requested_permission_list, activity_list, service_list, content_provider_list, broadcast_receiver_list, hardware_list, intentfilter_list
    except Exception as e:
        raise Exception("Fail to process xml file of apk {}:{}".format(apk_path, str(e)))


def get_feature_dex(apk_path, pmap, requested_permission_list):
    """
    get requested feature from .dex files
    :param apk_path: an absolute path of an apk
    :param pmap: PScout mapping
    :param requested_permission_list: a list of permissions
    :return: tupe of lists
    """
    used_permission_list = []
    restricted_api_list = []
    suspicious_api_list = []
    url_list = []
    try:
        apk_path = os.path.abspath(apk_path)
        a, dd, dx = AnalyzeAPK(apk_path)
    except Exception as e:
        raise Exception("Fail to load 'dex' files of apk {}:{} ".format(apk_path, str(e)))

    if not isinstance(dd, list):
        dd = [dd]  # may accommodate multiple dex files
    try:
        for i, d in enumerate(dd):
            for mtd in d.get_methods():
                dex_content = dx.get_method(mtd)
                for basic_block in dex_content.get_basic_blocks().get():
                    dalvik_code_list = []
                    for instruction in basic_block.get_instructions():
                        # dalvik code + performed body (api + arguments + return type)
                        code_line = instruction.get_name() + ' ' + instruction.get_output()
                        dalvik_code_list.append(code_line)
                    apis, suspicious_apis = get_specific_api(dalvik_code_list)
                    used_permissions, restricted_apis = get_permission_and_apis(apis,
                                                                                pmap,
                                                                                requested_permission_list,
                                                                                suspicious_apis)
                    used_permission_list.extend(used_permissions)
                    restricted_api_list.extend(restricted_apis)
                    suspicious_api_list.extend(suspicious_apis)

                    for code_line in dalvik_code_list:
                        url_search = re.search(
                            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            code_line,
                            re.IGNORECASE)
                        if url_search:
                            url = url_search.group()
                            url_domain = re.sub(r'(.*://)?([^/?]+).*', '\g<1>\g<2>', url)
                            url_list.append(url_domain)
        # remove duplication
        used_permission_list = list(set(used_permission_list))
        restricted_api_list = list(set(restricted_api_list))
        suspicious_api_list = list(set(suspicious_api_list))
        url_list = list(set(url_list))
        return used_permission_list, restricted_api_list, suspicious_api_list, url_list
    except Exception as e:
        raise Exception("Fail to process 'dex' files of apk {}:{}".format(apk_path, str(e)))


def get_specific_api(dalvik_code_list):
    """
    get invoked apis
    :param dalvik_code_list: a list of dalvik codes (line by line)
    :return: list of apis and list of suspicious apis
    """
    api_list = []
    suspicious_api_list = []

    for code_line in dalvik_code_list:
        if 'invoke-' in code_line:
            sub_parts = code_line.split(',')
            for part in sub_parts:
                if ';->' in part:
                    part = part.strip()
                    if part.startswith('Landroid'):
                        entire_api = part
                        api_parts = part.split(';->')
                        api_class = api_parts[0].strip()
                        api_name = api_parts[1].split('(')[0].strip()
                        api_dict = {'entire_api': entire_api, 'api_class': api_class, 'api_name': api_name}
                        api_list.append(api_dict)
                        if api_name in SuspiciousNames:
                            suspicious_api_list.append(api_class + '.' + api_name)
                for e in suspicious_api_list:
                    if e in part:
                        suspicious_api_list.append(e)

        for e in suspicious_api_list:
            if e in code_line:
                suspicious_api_list.append(e)

    # remove duplication
    suspicious_api_list = list(set(suspicious_api_list))

    return api_list, suspicious_api_list


def get_permission_and_apis(apis, pmap, requested_permission_list, suspicious_apis):
    """
    used permission and apis
    :param apis: a list of apis
    :param pmap: pscout mapping
    :param requested_permission_list: a list of permission
    :param suspicious_apis: a list of apis
    :return: used permission, restricted apis
    """
    used_permission_list = []
    restricted_api_list = []
    for api in apis:
        api_class = api['api_class'].replace('/', '.').replace("Landroid", "android").strip()
        permission = pmap.GetPermFromApi(api_class, api['api_name'])
        if permission is not None:
            if (permission in requested_permission_list) and (len(requested_permission_list) > 0):
                used_permission_list.append(permission)
                api_info = api_class + '.' + api['api_name']
                if api_info not in suspicious_apis:
                    restricted_api_list.append(api_info)
            else:
                api_info = api_class + '.' + api['api_name']
                if api_info not in suspicious_apis:
                    restricted_api_list.append(api_info)
    # remove duplication
    used_permission_list = list(set(used_permission_list))
    restricted_api_list = list(set(restricted_api_list))

    return used_permission_list, restricted_api_list


def dump_feature(new_path, data_dict):
    if not os.path.exists(os.path.dirname(new_path)):
        utils.mkdir(os.path.dirname(new_path))

    if not isinstance(data_dict, dict):
        raise TypeError("Not 'dict' format")

    with open(new_path, 'w') as f:
        for k, v in data_dict.items():
            for _v in v:
                f.write(str(k) + '_' + str(_v) + '\n')
    return


def load_feature(drebin_feature_path):
    """
    load feature for given path
    :rtype list
    """
    if os.path.isfile(drebin_feature_path):
        return utils.read_txt(drebin_feature_path)
    else:
        raise ValueError("Invalid path.")


def wrapper_load_features(path):
    try:
        return load_feature(path)
    except Exception as e:
        return e


class AxplorerMapping(object):
    def __init__(self):
        with open(os.path.join(current_dir, 'res/axplorerPermApi22Mapping.json'), 'rb') as FH:
            # Use SmallCase json file to prevent run time case conversion in GetPermFromApi
            import json
            self.PermApiDictFromJsonTemp = json.load(FH)
            self.PermApiDictFromJson = {}
            for Perms in self.PermApiDictFromJsonTemp:
                for Api in range(len(self.PermApiDictFromJsonTemp[Perms])):
                    ApiName = self.PermApiDictFromJsonTemp[Perms][Api][0].lower() + \
                              self.PermApiDictFromJsonTemp[Perms][Api][1].lower()
                    '''Exchange key and values inside the dictionary.'''
                    self.PermApiDictFromJson[ApiName] = Perms
        del self.PermApiDictFromJsonTemp

    def GetAllPerms(self):
        return list(self.PermApiDictFromJson.keys())

    def GetAllApis(self):
        return list(self.PermApiDictFromJson.values())

    def GetApisFromPerm(self, Perm):
        PermAsKey = Perm
        if PermAsKey not in self.PermApiDictFromJson:
            logger.error("Permission %s not found in the PScout Dict",
                         PermAsKey)
            return -1
        else:
            return self.PermApiDictFromJson[PermAsKey]

    def GetPermFromApi(self, ApiClass, ApiMethodName):
        ApiClass = ApiClass.lower()
        ApiMethodName = ApiMethodName.lower()

        ApiName = ApiClass + ApiMethodName
        if (ApiClass + ApiMethodName) in self.PermApiDictFromJson:
            return self.PermApiDictFromJson[ApiName]
        else:
            return None

    def PrintDict(self):
        pprint(self.PermApiDictFromJson)

    def PrintAllPerms(self):
        for PermAsKey in self.PermApiDictFromJson:
            print(PermAsKey)

    def PrintAllApis(self):
        for Api in self.PermApiDictFromJson.values():
            print(Api)

    def PrintApisForPerm(self, Perm):
        PermAsKey = Perm

        if PermAsKey not in self.PermApiDictFromJson:
            warnings.warn("Permission {} not found in the PScout Dict".format(
                PermAsKey))
            return -1

        for Api in self.PermApiDictFromJson[Perm]:
            pprint(Api)
        return 0

    ##################################################
    #                 Sorting the dict               #
    ##################################################
    def SortDictByKeys(self):
        self.PermApiDictFromJson = \
            collections.OrderedDict(sorted(self.PermApiDictFromJson.items()))
