import torch
import yaml
from pathlib import Path

DATA_PATH = Path('data')
CONFIGS_PATH = Path('configs')
MODELS_PATH = Path('models')
MAIN_CONFIG_PATH = CONFIGS_PATH / "model_config.yml"

RAW_DATA = DATA_PATH / "raw"
PROCESSED_DATA = DATA_PATH / "processed"
FASTTEXT_MODEL = Path("..") / "kgr10.plain.skipgram.dim300.neg10.bin"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_CUDA = torch.cuda.is_available()
LONGEST_SAMPLE = 196

config = yaml.safe_load(open(MAIN_CONFIG_PATH))
PAD_TOKEN = config['tagset_size']

EPSILON = 1e-8
