# coding: utf-8
import os
import sys

sys.path.append(os.path.normpath('../../src'))
sys.path.append(os.path.normpath('../../src/IncrementalKS'))
from detectors.lcode import LCODE_WD
from detectors.md3 import MD3
from detectors.mwu import MWU
from detectors.iks import IKS_
from detectors.hdddm import HDDDM
from detectors.acc_based import ADWIN_, DDM_, HDDM_A_, HDDM_W_, KSWIN_, PageHinkley_
from detectors.baselines import W_UPDATE, WO_UPDATE
from experiment_stream import XPS_BASE, XPS_LCODE, XPS_MD3, XPS_UPDATE

WORK_DIR = os.path.normpath('../..')
BASE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'base')
EXTERNAL_DATA_DIR = os.path.join(WORK_DIR, 'data', 'external')
INTERIM_DATA_DIR = os.path.join(WORK_DIR, 'data', 'interim')
SAVE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed_stream1')
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')

BASE_DATA_TYPE = 'cat'

OUR_DETECTORS = {
    'lcode_wd': LCODE_WD,
}

SV_DETECTORS = {
    'ph': PageHinkley_,
    'adwin': ADWIN_,
    'kswin': KSWIN_,
    'ddm': DDM_,
    'hddm_a': HDDM_A_,
    'hddm_w': HDDM_W_,
}

USV_DETECTORS = {
    'md3': MD3,
    'mwu': MWU,
    'iks': IKS_,
    'hdddm': HDDDM,
}

REFERENCES = {
    'no_update': WO_UPDATE,
    'update': W_UPDATE,
}

ALL_DETECTORS = {
    **OUR_DETECTORS,
    **USV_DETECTORS,
    **SV_DETECTORS,
    **REFERENCES,
}

DETECTOR_TYPES = {
    'lcode_wd': 'Ours',
    'lcode_hd': 'Ours',
    'lcode_ks': 'Ours',
    'md3': 'Unsupervised methods',
    'mwu': 'Unsupervised methods',
    'iks': 'Unsupervised methods',
    'hdddm': 'Unsupervised methods',
    'ph': 'Supervised methods',
    'adwin': 'Supervised methods',
    'kswin': 'Supervised methods',
    'ddm': 'Supervised methods',
    'hddm_a': 'Supervised methods',
    'hddm_w': 'Supervised methods',
    'no_update': 'Reference standard',
    'update': 'Reference standard',
}

COLORS = {
    'Ours': '#377eb8',
    'Unsupervised methods': '#ff7f00',
    'Supervised methods': '#4daf4a',
    'Reference standard': '#f781bf',
}

DATASETS = [
    'Insects_A',
    'Insects_I',
    'Insects_IG',
    'Insects_IR',
    'Insects_IAR',
    'Insects_A_imb',
    'Insects_I_imb',
    'Insects_IG_imb',
    'Insects_IR_imb',
    'Insects_IAR_imb',
    'Insects_OOC'
]

EX_RUNNERS = {
    'lcode_wd': XPS_LCODE,
    'md3': XPS_MD3,
    'mwu': XPS_BASE,
    'iks': XPS_BASE,
    'hdddm': XPS_BASE,
    'ph': XPS_BASE,
    'adwin': XPS_BASE,
    'kswin': XPS_BASE,
    'ddm': XPS_BASE,
    'hddm_a': XPS_BASE,
    'hddm_w': XPS_BASE,
    'no_update': XPS_BASE,
    'update': XPS_UPDATE,
}

DRIFT_POINTS = {
    'Insects_A': [14352, 19500, 33240, 38682, 39510], 
    'Insects_I': [],
    'Insects_IG': [14028],
    'Insects_IR': [26568, 53364],
    'Insects_IAR': [26568, 53364],
    'Insects_A_imb': [83859, 128651, 182320, 242883, 268380], 
    'Insects_I_imb': [],
    'Insects_IG_imb': [58159],
    'Insects_IR_imb': [150683, 301365],
    'Insects_IAR_imb': [150683, 301365],
    'Insects_OOC': [],
}

DRIFT_TYPES = {
    'Insects_A': 'Abrupt',
    'Insects_I': 'Incremental',
    'Insects_IG': 'Incremental, Gradual',
    'Insects_IR': 'Incremental, Reoccurring',
    'Insects_IAR': 'Incremental, Abrupt, Reoccurring',
    'Insects_A_imb': 'Abrupt',
    'Insects_I_imb': 'Incremental',
    'Insects_IG_imb': 'Incremental, Gradual',
    'Insects_IR_imb': 'Incremental, Reoccurring',
    'Insects_IAR_imb': 'Incremental, Abrupt, Reoccurring',
    'Insects_OOC': 'Out of control',
}

DETECTOR_PRMSS = {
    'lcode_wd': {'gamma': [0.002], 'alpha': [2]},
#     'lcode_wd': {'gamma': [0.002, 0.005, 0.01], 'alpha': [2, 3]},
    'md3': {'theta': [2]},
    'mwu': {'alpha': [0.01]},
    'iks': {'alpha': [0.001]},
    'hdddm': {'gamma': [1]},
    'ph': {'threshold': [50]},
    'adwin': {'delta': [0.002]},
    'kswin': {'alpha': [0.005]},
    'ddm': {'out_control_level': [3]},
    'hddm_a': {'drift_confidence': [0.001]},
    'hddm_w': {'drift_confidence': [0.001]},
    'no_update': {},
    'update': {},
}

DETECTOR_PRMS_MAP = {
    'lcode_wd': {'gamma': [0.002], 'alpha': [100]},
}