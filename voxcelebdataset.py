from dataclasses import dataclass, fields
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import skimage.io


@dataclass
class VoxCelebUtterance:
    identity: str
    reference: str
    offset: int
    fv_conf: float
    asd_conf: float
    utterance_name: str
    fps: int
    utterance_start_time: float
    utterance_end_time: float
    spectrogram_time_duration: float
    num_spectrograms: int

    @classmethod
    def from_txt(cls, txt_file: Path, spectrogram_time_duration: float=3.0, fps: int=25):
        inst = {
            'utterance_name' : txt_file.stem,
            'spectrogram_time_duration' : spectrogram_time_duration
        }
        with open(txt_file, 'r') as f:
            if (txt_file.parent.name == 'utrA-v8pPm4'):
                print()
            for field in fields(cls)[:5]:
                inst[field.name] = field.type(next(f).split()[-1])

            # Blank line and header
            line = next(f).strip()
            line = next(f).strip()

            utterance_start_frame = -1
            utterance_end_frame = -1
            for line in f:
                entries = line.strip().split()
                if len(entries) >= 5:
                    frame, x, y, w, h = [int(entry) for entry in entries[:5]]
                    if utterance_start_frame == -1:
                        utterance_start_frame = frame
                    utterance_end_frame = frame

        utterance_start_time = utterance_start_frame / fps
        utterance_end_time = utterance_end_frame / fps
        num_spectrograms = int((utterance_end_time - utterance_start_time) / spectrogram_time_duration)

        inst['fps'] = fps
        inst['utterance_start_time'] = utterance_start_time
        inst['utterance_end_time'] = utterance_end_time
        inst['num_spectrograms'] = num_spectrograms

        return cls(**inst)

def get_first_if_tuple(obj):
    if type(obj) is tuple:
        return obj[0]
    return obj

@dataclass
class VoxCelebUtterances:
    identity: str
    url_md5: str
    utterances_dir: Path
    utterances: list[VoxCelebUtterance]
    total_spectrograms: int

    @classmethod
    def from_celeb_url_md5_dir(cls, utterances_dir: Path):
        inst = {
            'identity' : utterances_dir.parent.name,
            'url_md5' : utterances_dir.name,
            'utterances_dir' : utterances_dir,
        }

        total_spectrograms = 0
        utterances = []
        for f in utterances_dir.iterdir():
            if not f.is_file():
                continue
            utterances += [VoxCelebUtterance.from_txt(f)]
            total_spectrograms += utterances[-1].num_spectrograms
        inst['total_spectrograms'] = total_spectrograms
        inst['utterances'] = utterances

        return cls(**inst)

def image_to_uint8(img):
    img_range = img.max() - img.min()
    if img_range == 0:
        return np.full_like(img, 127, dtype=np.uint8)
    return (255 * (img - img.min()) / img_range).astype(np.uint8)

def generate_voxceleb_spectrograms(celeb_utterances: list[VoxCelebUtterances], audio_file: Path, output_dir: Path) -> bool:
    sampling_rate = 16000
    audio, sampling_rate = librosa.load(audio_file, sr=sampling_rate, mono=True)
    if audio is None:
        print(f'Could not load audio {audio_file=}')
        return False
    seconds_to_samples = lambda seconds: int(seconds * sampling_rate)

    # Spectrogram parameters
    n_fft = 2048
    hop_length = seconds_to_samples(0.01)

    success = True
    for voxceleb_utterances in celeb_utterances:
        spectrogram_output_dir = output_dir / voxceleb_utterances.identity / voxceleb_utterances.url_md5
        if not spectrogram_output_dir.exists():
            spectrogram_output_dir.mkdir()

        utterances = voxceleb_utterances.utterances
        if len(utterances) == 0:
            continue

        for voxceleb_utterance in utterances:
            samples_per_spectrogram = seconds_to_samples(voxceleb_utterance.spectrogram_time_duration)
            utterance_start_sample = seconds_to_samples(voxceleb_utterance.utterance_start_time)
            for spectrogram_idx in range(voxceleb_utterance.num_spectrograms):
                start_sample = utterance_start_sample + samples_per_spectrogram * spectrogram_idx
                audio_utterance = audio[start_sample:start_sample + samples_per_spectrogram]
                if len(audio_utterance) != samples_per_spectrogram:
                    # Bad label outside bounds of audio?
                    print(f'Tried to get audio outside of array bounds: {len(audio_utterance)=}, {start_sample=}, {samples_per_spectrogram=}')
                    success = False
                    continue

                mel_power_spectrogram = librosa.feature.melspectrogram(y=audio_utterance, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
                mel_db_spectrogram = librosa.power_to_db(mel_power_spectrogram, ref=np.max)
                # librosa.display.specshow(mel_db_spectrogram, x_axis='s', y_axis='mel', sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
                mel_db_spectrogram = np.flip(mel_db_spectrogram, axis=0)
                skimage.io.imsave(spectrogram_output_dir / f'{voxceleb_utterance.utterance_name}_{spectrogram_idx}.png', image_to_uint8(mel_db_spectrogram))

    return success
