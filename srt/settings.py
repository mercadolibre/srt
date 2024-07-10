from srt import fs

here = fs.parent(__file__)
data_dir = fs.ensure_exists(fs.abspath(fs.join(here, '../data')))

fs.ensure_exists(fs.join(data_dir, 'input'))
fs.ensure_exists(fs.join(data_dir, 'intermediate'))
fs.ensure_exists(fs.join(data_dir, 'output'))
TMP_DIR = fs.ensure_exists(fs.join(data_dir, 'tmp'))

ONLY_FEW = False

RAW_RATINGS_FNAME = fs.join(data_dir, 'input/all_csv_files.csv')
RAW_BEAUTY_FNAME = fs.join(data_dir, 'input/raw_beauty.jl.gz')
RAW_SPORTS_FNAME = fs.join(data_dir, 'input/raw_sports.jl.gz')
RAW_METADATA_FNAME = fs.join(data_dir, 'input/metadata.jl.gz')

PARTITIONED_RATINGS_DIR = fs.join(data_dir, 'intermediate/ratings')
PARTITIONED_METADTA_DIR = fs.join(data_dir, 'intermediate/metadata')
POST_PROCESSED_RATINGS_DIR = fs.join(data_dir, 'intermediate/clean_ratings')

TOKENIZED_METADATA_DIR = fs.join(data_dir, 'intermediate/tokenized_metadata')

WARM_USER_SET_FNAME = fs.join(data_dir, 'intermediate/warm_users.pkl')
SCALED_DT_DIR = fs.join(data_dir, 'intermediate/scaled_ratings')
SENTENCE_PIECE_DIR = fs.join(data_dir, 'intermediate/sentence_piece')
RECBOLE_DATASETS = fs.join(data_dir, 'output')

RECBOLE_DIR = fs.abspath(fs.join(here, '../MyRecbole'))
