#!/usr/bin/env python

import argparse
import voxcelebdownloader
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SpeakerIdentificationMetaLearning',
        description='Perform speaker identification using meta learning on the voxceleb dataset',
    )
    subparsers = parser.add_subparsers(dest='command')

    # subparser for downloading voxceleb data from youtube (takes a long time)
    download_parser = subparsers.add_parser('voxceleb-download', help='Download the voxceleb dataset')
    download_parser.add_argument(
        '--url-directory',
        type=str,
        required=True, 
        help='Voxceleb "URLs and timestamps" directory'
    )
    download_parser.add_argument(
        '--audio-directory',
        type=str,
        required=True, 
        help='Directory to store downloaded audio files (temporary directory if --keep-audio isn\'t passed)'
    )
    download_parser.add_argument(
        '--output-directory',
        type=str,
        required=True, 
        help='Directory to store labeled spectrogram images from the utterances'
    )
    download_parser.add_argument(
        '--keep-audio',
        required=False,
        action='store_true',
        help='Full audio files will be kept after spectrogram images have been created'
    )
    download_parser.add_argument(
        '--force-post-process',
        required=False,
        action='store_true',
        help='Forces the post processing to occur, even if existing data is already found'
    )

    args = parser.parse_args()
    if args.command == 'voxceleb-download':
        voxcelebdownloader.voxceleb_download(Path(args.url_directory), Path(args.audio_directory), Path(args.output_directory), args.force_post_process, args.keep_audio)
