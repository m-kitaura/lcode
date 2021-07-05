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
from experiment_cost import XP_COST

WORK_DIR = os.path.normpath('../..')
BASE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'base')
EXTERNAL_DATA_DIR = os.path.join(WORK_DIR, 'data', 'external')
INTERIM_DATA_DIR = os.path.join(WORK_DIR, 'data', 'interim')
SAVE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed_cost1')
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


ALL_DETECTORS = {
    **OUR_DETECTORS,
    **USV_DETECTORS,
    **SV_DETECTORS,
}

DETECTOR_TYPES = {
    'lcode_wd': 'Ours',
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
#     'Insects_A_imb',
#     'Insects_I_imb',
#     'Insects_IG_imb',
#     'Insects_IR_imb',
#     'Insects_IAR_imb',
#     'Insects_OOC'
]

EX_RUNNERS = {
    'lcode_wd': XP_COST,
    'md3': XP_COST,
    'mwu': XP_COST,
    'iks': XP_COST,
    'hdddm': XP_COST,
    'ph': XP_COST,
    'adwin': XP_COST,
    'kswin': XP_COST,
    'ddm': XP_COST,
    'hddm_a': XP_COST,
    'hddm_w': XP_COST,
}

DETECTOR_PRMSS = {
    'lcode_wd': {'gamma': [0.002], 'alpha': [1e10]},
    'md3': {'theta': [1e10]},
    'mwu': {'alpha': [0]},
    'iks': {'alpha': [0]},
    'hdddm': {'gamma': [1e10]},
    'ph': {'threshold': [1e10]},
    'adwin': {'delta': [0]},
    'kswin': {'alpha': [0]},
    'ddm': {'out_control_level': [1e10]},
    'hddm_a': {'drift_confidence': [1e-30]},
    'hddm_w': {'drift_confidence': [1e-30]},
}
