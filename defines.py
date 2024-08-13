from enum import Enum

# Constants file

# Categories
CATEGORIES = ['staff']

CATEGORIES_TO_NUM = dict({
  'background': 0,
  'staff': 1,
  'empty-staff': 1,
})
NUM_TO_CATEGORIES = dict({
  0: 'background',
  1: 'staff',
})


# Datasets
DATASETS = [
  'Capitan',
  'SEILS',
  'FMT_C',
]


DEBUG_FLAG = False


BBOX_REDIMENSION = 0.8
BBOX_REDIMENSIONED_RECOVER = 1 / BBOX_REDIMENSION

TRAIN_NUM_EPOCHS = 100
TRAIN_PATIENTE = 20

DRIVE_IA_FOLDER = ''
DRIVE_IA_FOLDER = '/content/drive/MyDrive/TFM/IA'

DRIVE_DATASETS_FOLDER = f'{DRIVE_IA_FOLDER}/datasets'
DRIVE_MODELS_FOLDER = f'{DRIVE_IA_FOLDER}/models'
DRIVE_LOGS_FOLDER = f'{DRIVE_IA_FOLDER}/logs'
DRIVE_IMG_FOLDER = f'{DRIVE_IA_FOLDER}/img'

DRIVE_TRAIN_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/train'
DRIVE_VAL_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/val'
DRIVE_TEST_LOGS_FOLDER = f'{DRIVE_LOGS_FOLDER}/test'

DRIVE_VAL_IMG_FOLDER = f'{DRIVE_IMG_FOLDER}/val'
DRIVE_TEST_IMG_FOLDER = f'{DRIVE_IMG_FOLDER}/test'

SAE_IMAGE_SIZE =  (512, 512)

BIN_UMBRALS = [i/100 for i in range(10, 91, 5)]
DROPOUT_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

class PredictionsCombinationType(Enum):
    NONE = ''
    MEAN = 'mean'
    MAX  = 'max'
