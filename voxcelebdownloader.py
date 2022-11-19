import os
import shutil
import ray
import timeit
import argparse
from pathlib import Path

import voxcelebdataset


def voxceleb_download(url_directory: Path, audio_directory: Path, output_directory: Path, force_post_process: bool, keep_audio: bool) -> None:
    celeb_directory = url_directory / 'txt'

    # url_md5 --> list of celebs that are labeled in the audio and need to be processed
    url_md5_to_celebs = {}
    for celeb_dir in celeb_directory.iterdir():
        celeb_output_dir = output_directory / celeb_dir.name
        if not celeb_output_dir.exists():
            celeb_output_dir.mkdir()

        for celeb_url_md5_dir in celeb_dir.iterdir():
            url_md5 = celeb_url_md5_dir.name
            utterances = voxcelebdataset.VoxCelebUtterances.from_celeb_url_md5_dir(celeb_url_md5_dir)
            celeb_url_md5_spectrograms_dir = celeb_output_dir / url_md5
            if celeb_url_md5_spectrograms_dir.exists():
                spectrograms_already_generated = len(os.listdir(celeb_url_md5_spectrograms_dir)) >= utterances.total_spectrograms
                if force_post_process or not spectrograms_already_generated:
                    shutil.rmtree(celeb_url_md5_spectrograms_dir)

            if not celeb_url_md5_spectrograms_dir.exists():
                if url_md5 not in url_md5_to_celebs:
                    url_md5_to_celebs[url_md5] = [utterances]
                else:
                    url_md5_to_celebs[url_md5] += [utterances]

    download_tasks = []
    for url_md5, celebs_utterances in url_md5_to_celebs.items():
        download_tasks += [process_voxceleb_utterances.remote(celebs_utterances, audio_directory, output_directory, keep_audio=keep_audio)]
        # download_task = process_voxceleb_utterances(celebs_utterances, audio_directory, output_directory, keep_audio=keep_audio)

    print(f'Beginning parallel download and processing of {len(download_tasks)} audio files...')
    results = ray.get(download_tasks)
    print(f'Finished voxceleb download and processing, succeeded on {sum(results)}/{len(download_tasks)} audio files ({sum(results)/len(download_tasks)})')


@ray.remote
def process_voxceleb_utterances(celebs_utterances: list[voxcelebdataset.VoxCelebUtterances], audio_dir: Path, output_dir: Path, audio_format: str='wav', keep_audio=False) -> None:
    # before doing the download (which can take a while), check if we already have it
    url_md5 = celebs_utterances[0].url_md5
    audio_path = audio_dir / f'{url_md5}.{audio_format}'
    downloaded = False
    already_downloaded = audio_path.exists()
    if not already_downloaded:
        # download the video associated with url_md5 to the audio_dir
        downloaded = download_wav_for_youtube_md5(url_md5, output_dir=audio_dir)

    success = already_downloaded or downloaded
    if success:
        # create spectrograms
        success = voxcelebdataset.generate_voxceleb_spectrograms(celebs_utterances, audio_path, output_dir)
        if not keep_audio:
            os.remove(audio_path)

    print(f'Result of processing {len(celebs_utterances)} celeb for {url_md5=} --> {already_downloaded=}, {downloaded=}, {success=}')
    return success


def download_wav_for_youtube_md5(url_md5, audio_format='wav', output_dir=None) -> bool:
    start_time = timeit.default_timer()
    file_size = 0
    file_name = f'{url_md5}.{audio_format}'
    ret = os.system(f"youtube-dl https://www.youtube.com/watch\?v\={url_md5} --extract-audio --audio-format {audio_format} -o '{url_md5}.%(ext)s' --quiet")
    if ret == 0 and output_dir is not None:
        file_size = os.path.getsize(file_name)
        if output_dir is not None:
            shutil.move(file_name, output_dir)
    time = timeit.default_timer() - start_time
    print(f'Download of {file_name} completed: {ret=}, {file_size=}, {time=}')
    return ret == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Downloads the voxceleb data')
    parser.add_argument(
        '--url-directory',
        type=str,
        required=True, 
        help='Voxceleb "URLs and timestamps" directory'
    )
    parser.add_argument(
        '--audio-directory',
        type=str,
        required=True, 
        help='Directory to store downloaded audio files (temporary directory if --keep-audio isn\'t passed)'
    )
    parser.add_argument(
        '--output-directory',
        type=str,
        required=True, 
        help='Directory to store labeled spectrogram images from the utterances'
    )
    parser.add_argument(
        '--keep-audio',
        required=False,
        action='store_true',
        help='Full audio files will be kept after spectrogram images have been created'
    )
    parser.add_argument(
        '--force-post-process',
        required=False,
        action='store_true',
        help='Forces the post processing to occur, even if existing data is already found'
    )

    args = parser.parse_args()
    voxceleb_download(Path(args.url_directory), Path(args.audio_directory), Path(args.output_directory), args.force_post_process, args.keep_audio)
