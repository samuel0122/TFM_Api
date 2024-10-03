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
  VOTES = 'votes'

class ForwardParameters:
  def __init__(
      self,
      uses_redimension_vertical: bool = True,
      uses_redimension_horizontal: bool = True,
      bin_umbral: float = 0.5,
      train_dropout: float = 0.0,
      val_dropout: float = 0.0,
      times_pass_model: int = 1,
      type_combination: PredictionsCombinationType = PredictionsCombinationType.NONE,
      votes_threshold: float = 0.5
    ):
      self.uses_redimension_vertical = uses_redimension_vertical
      self.uses_redimension_horizontal = uses_redimension_horizontal
      self.bin_umbral = bin_umbral
      self.train_dropout = train_dropout
      self.val_dropout = val_dropout
      self.times_pass_model = times_pass_model
      self.type_combination = type_combination
      self.votes_threshold = votes_threshold

# Datasets
DATASETS = [
  'Capitan',
  'SEILS',
  'FMT_C',
]

MODEL_FORWARD_PARAMETERS = [ 
  ForwardParameters(
    uses_redimension_vertical=True,
    uses_redimension_horizontal = True,
    bin_umbral=0.45, #0.45
    train_dropout=0.3,
    val_dropout=0, #0.3
    times_pass_model=1, #63
    type_combination=PredictionsCombinationType.NONE, #PredictionsCombinationType.VOTES
    votes_threshold = 0.5
  ),
  ForwardParameters(
    uses_redimension_vertical=True,
    uses_redimension_horizontal = True,
    bin_umbral=0.6, #0.6
    train_dropout=0.2,
    val_dropout=0, #0.1
    times_pass_model=1, #31
    type_combination=PredictionsCombinationType.NONE, #PredictionsCombinationType.VOTES
    votes_threshold = 0.5
  ),
  ForwardParameters(
    uses_redimension_vertical=True,
    uses_redimension_horizontal = True,
    bin_umbral=0.5, #0.5
    train_dropout=0.2,
    val_dropout=0, #0.2
    times_pass_model=1, #31
    type_combination=PredictionsCombinationType.NONE, #PredictionsCombinationType.MEAN
    votes_threshold = 0.5
  )
]

# Carpetas de imágenes
API_IMAGES_FOLDER = 'static/images'
API_IMAGES_FOLDERS = {
    'capitan': f'{API_IMAGES_FOLDER}/Capitan',
    'seils': f'{API_IMAGES_FOLDER}/SEILS',
    'fmt_c': f'{API_IMAGES_FOLDER}/FMT_C'
}