import argparse

from srt.scaling_datasets.beauty_and_sports import build_beauty_and_sports_datasets
from srt.scaling_datasets.download import download_raw_files, partition_big_files
from srt.scaling_datasets.post_process_meta import train_sentence_piece, tokenize_metadata
from srt.scaling_datasets.post_process_ratings import post_process_ratings
from srt.scaling_datasets.scaled_datasets import build_scaling_datasets


def run_all():
    download_raw_files()

    partition_big_files()

    train_sentence_piece()
    tokenize_metadata()
    
    post_process_ratings()
    build_scaling_datasets()

    build_beauty_and_sports_datasets()


steps = {
    'download_raw_files': download_raw_files,
    'partition_big_files': partition_big_files,

    'train_sentence_piece': train_sentence_piece,
    'tokenize_metadata': tokenize_metadata,

    'post_process_ratings': post_process_ratings,
    'build_scaling_datasets': build_scaling_datasets,
    'build_beauty_and_sports_datasets': build_beauty_and_sports_datasets,
    'all': run_all
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=steps.keys(), required=True)

    args = parser.parse_args()
    steps[args.step]()


if __name__ == '__main__':
    main()
