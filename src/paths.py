from pathlib import Path

from regex import P


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

MACO_DATASET_DIR = DATA_DIR / "MECO" / "tabular_en"
ROOT_RUNS_DIR = ROOT_DIR / "data" / "runs"
IMAGES_MACO_DIR = DATA_DIR / "MECO" / "texts_en_images"

TEXTS_DF_PATH = DATA_DIR / "MECO" / "texts_en" / "sentences_by_char.csv"
WORDS_SURPS_PATH = DATA_DIR / "MECO" / "surprisals_en" / "word_surprisal_meco.csv"
CHARACTER_SURPS_PATH = DATA_DIR / "MECO" / "surprisals_en" / "char_surprisal_meco.csv"
