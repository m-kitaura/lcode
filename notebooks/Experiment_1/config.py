# coding: utf-8
import os
import sys
import json

sys.path.append(os.path.normpath('../../src'))
sys.path.append(os.path.normpath('../../src/IncrementalKS'))
from detectors.lcode import LCODE_WD, LCODE_HD, LCODE_KS
from detectors.md3 import MD3
from detectors.mwu import MWU
from detectors.iks import IKS_
from detectors.hdddm import HDDDM
from detectors.acc_based import ADWIN_, DDM_, HDDM_A_, HDDM_W_, KSWIN_, PageHinkley_
from detectors.baselines import W_UPDATE, WO_UPDATE
from experiment import XP_BASE, XP_LCODE, XP_MD3

WORK_DIR = os.path.normpath('../..')
BASE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'base')
EXTERNAL_DATA_DIR = os.path.join(WORK_DIR, 'data', 'external')
INTERIM_DATA_DIR = os.path.join(WORK_DIR, 'data', 'interim')
DRIFT_INFO_DIR = os.path.join(WORK_DIR, 'data', 'info')
SAVE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed1')
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')

BASE_DATA_TYPE = 'shuffled'

OUR_DETECTORS = {
    'lcode_wd': LCODE_WD,
    'lcode_hd': LCODE_HD,
    'lcode_ks': LCODE_KS,
}

USV_DETECTORS = {
    'md3': MD3,
    'mwu': MWU,
    'iks': IKS_,
    'hdddm': HDDDM,
}

SV_DETECTORS = {
    'ph': PageHinkley_,
    'adwin': ADWIN_,
    'kswin': KSWIN_,
    'ddm': DDM_,
    'hddm_a': HDDM_A_,
    'hddm_w': HDDM_W_,
}

REFERENCES = {
    'no_update': WO_UPDATE,
    'update': W_UPDATE,
}

ALL_USV_DETECTORS = {
    **OUR_DETECTORS,
    **USV_DETECTORS,
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

All_DATASETS = ['Anuran_Calls', 'avila', 'Cardiotocography', 'credit card', 'EEG', 'HTRU2',
                'image', 'landsat', 'letter-recognition', 'magic', 'occupancy', 'page-blocks',
                'pendigits', 'shuttle', 'Wall-following', 'wifi_localization', 'wine']
if os.path.exists('dataset.conf'):
    with open('dataset.conf', 'r') as f:
        dataset_conf = json.load(f)
        DATASETS = dataset_conf['selected_datasets']

DATA_PRMS = {
    'seed': list(range(20)),
    'how_feature': ['relevant', 'irrelevant'],
    'how_class': ['minor']
}

EX_RUNNERS = {
    'lcode_wd': XP_LCODE,
    'lcode_hd': XP_LCODE,
    'lcode_ks': XP_LCODE,
    'md3': XP_MD3,
    'mwu': XP_BASE,
    'iks': XP_BASE,
    'hdddm': XP_BASE,
    'ph': XP_BASE,
    'adwin': XP_BASE,
    'kswin': XP_BASE,
    'ddm': XP_BASE,
    'hddm_a': XP_BASE,
    'hddm_w': XP_BASE,
    'no_update': XP_BASE,
    'update': XP_BASE,
}

DETECTOR_PRMSS = {
    'lcode_wd': {'gamma': [0.002, 0.005, 0.01], 'alpha': [2, 3]},
    'lcode_hd': {'gamma': [0.002, 0.005, 0.01], 'alpha': [2, 3]},
    'lcode_ks': {'gamma': [0.002, 0.005, 0.01], 'alpha': [2, 3]},
    'md3': {'theta': [0.5, 1, 2, 3, 4, 5]},
    'mwu': {'alpha': [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]},
    'iks': {'alpha': [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]},
    'hdddm': {'gamma': [0.2, 0.5, 1, 1.5, 2, 3]},
    'ph': {'threshold': [10, 20, 50, 100, 200, 500]},
    'adwin': {'delta': [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]},
    'kswin': {'alpha': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]},
    'ddm': {'out_control_level': [1, 2, 3, 6, 12, 20]},
    'hddm_a': {'drift_confidence': [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]},
    'hddm_w': {'drift_confidence': [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]},
    'no_update': {},
    'update': {},
}

# weight of a false alarm to decide the best balanced parameters
FA_WEIGHT = 0.1

if os.path.exists('detector.conf'):
    with open('detector.conf', 'r') as f:
        DETECTOR_PRMS = json.load(f)
