# coding: utf-8
import os

WORK_DIR = os.path.normpath('../..')
BASE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'base')
EXTERNAL_DATA_DIR = os.path.join(WORK_DIR, 'data', 'external')
INTERIM_DATA_DIR = os.path.join(WORK_DIR, 'data', 'interim')
DRIFT_INFO_DIR = os.path.join(WORK_DIR, 'data', 'info')
SAVE_DATA_DIR = os.path.join(WORK_DIR, 'data', 'processed_prm1')
OUTPUT_DIR = os.path.join(WORK_DIR, 'output')

BASE_DATA_TYPE = 'shuffled'

DATA_PRMS = {
    'seed': list(range(10)),
    'how_feature': ['relevant', 'irrelevant'],
    'how_class': ['minor']
}

LCODE_PRMS = {
    'gamma': [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    'alpha': [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
}
