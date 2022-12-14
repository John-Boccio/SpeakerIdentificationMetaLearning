from dataclasses import dataclass, fields
from pathlib import Path
import librosa
import librosa.display
import numpy as np
import skimage.io
import glob
import torch
from torchvision import transforms
from torch.utils.data import dataset, sampler, dataloader
from PIL import Image


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


@dataclass
class VoxCelebUtterances:
    identity: str
    url_md5: str
    utterances_dir: Path
    utterances: list
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


def generate_voxceleb_spectrograms(celeb_utterances: list, audio_file: Path, output_dir: Path) -> bool:
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


def get_voxceleb_stats():
    pass


class VoxCelebDataset(dataset.Dataset):
    def __init__(self, voxceleb_dir: list, spectrograms_dir: Path, dynamic_load=False, num_support: int=-1, num_query: int=-1, resize: tuple=None, seed: int=None) -> None:
        super().__init__()

        self._dynamic_load = dynamic_load
        self._num_support = num_support
        self._num_query = num_query
        self._use_tasks = (num_support != -1 and num_query != -1)
        t = [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.type(torch.FloatTensor)),
        ]
        if resize is not None:
            t += [transforms.Resize(resize)]
        self._transform = transforms.Compose(t)

        self._rng = np.random.default_rng(seed)
        num_dirs = len([f.is_dir() for f in voxceleb_dir.iterdir()])
        if self._use_tasks:
            self.celebs = []
            self.celeb_spectrograms = {}
            for i, celeb_dir in enumerate(voxceleb_dir.iterdir()):
                if (i % 100) == 0 or (i+1) == num_dirs:
                    print(f'Loading celeb dir {i+1}/{num_dirs}')
                celeb_utterances = spectrograms_dir / celeb_dir.stem
                celeb_spectrograms = glob.glob(str(celeb_utterances / '*' / '*.png'))
                if celeb_spectrograms:
                    if len(celeb_spectrograms) >= (num_support + num_query):
                        self.celebs += [celeb_dir.name]
                        if self._dynamic_load:
                            self.celeb_spectrograms[celeb_dir.name] = celeb_spectrograms
                        else:
                            self.celeb_spectrograms[celeb_dir.name] = [self._transform(Image.open(spectrogram_path)) for spectrogram_path in celeb_spectrograms]
                    else:
                        print(f'Dropping celeb {celeb_dir.name} due to only having {len(celeb_spectrograms)} spectrograms ({num_support=}, {num_query=}, required={num_support + num_query})')
            self._rng.shuffle(self.celebs)
        else:
            self.spectrograms = []
            self.labels = []
            for i, celeb_dir in enumerate(voxceleb_dir.iterdir()):
                if (i % 100) == 0 or (i+1) == num_dirs:
                    print(f'Loading celeb dir {i+1}/{num_dirs}')
                celeb_utterances = spectrograms_dir / celeb_dir.stem
                celeb_spectrograms = glob.glob(str(celeb_utterances / '*' / '*.png'))
                if celeb_spectrograms:
                    celeb_id = int(celeb_dir.name[2:])
                    if self._dynamic_load:
                        self.spectrograms += celeb_spectrograms
                    else:
                        self.spectrograms += [self._transform(Image.open(spectrogram_path)) for spectrogram_path in celeb_spectrograms]
                    self.labels += [celeb_id for _ in celeb_spectrograms]

    def __getitem__(self, idxs):
        if self._use_tasks:
            celeb_idxs = idxs
            spectrograms_support, spectrograms_query = [], []
            labels_support, labels_query = [], []

            for label, celeb_idx in enumerate(celeb_idxs):
                celeb_spectrograms = self.celeb_spectrograms[self.celebs[celeb_idx]]
                spectrogram_idxs = self._rng.choice(len(celeb_spectrograms), size=(self._num_query + self._num_support), replace=False)
                if self._dynamic_load:
                    spectrograms = [self._transform(Image.open(celeb_spectrograms[spectrogram_idx])) for spectrogram_idx in spectrogram_idxs]
                else:
                    spectrograms = celeb_spectrograms[spectrogram_idxs]
                spectrograms_support += spectrograms[:self._num_support]
                spectrograms_query += spectrograms[self._num_support:]
                labels_support += [label] * self._num_support
                labels_query += [label] * self._num_query

            spectrograms_support = torch.stack(spectrograms_support)
            labels_support = torch.tensor(labels_support)
            spectrograms_query = torch.stack(spectrograms_query)
            labels_query = torch.tensor(labels_query)
            return spectrograms_support, labels_support, spectrograms_query, labels_query
        else:
            if self._dynamic_load:
                spectrogram = self._transform(Image.open(self.spectrograms[idxs]))
            else:
                spectrogram = self.spectrograms[idxs]
            return spectrogram, self.labels[idxs]


class VoxCelebTaskSampler(sampler.Sampler):
    def __init__(self, num_celebs, num_way, num_tasks, seed=None) -> None:
        super().__init__(None)
        self._num_celebs = num_celebs
        self._num_way = num_way
        self._num_tasks = num_tasks
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        return (
            self._rng.choice(self._num_celebs, size=self._num_way, replace=False) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


class VoxCelebSpectrogramSampler(sampler.Sampler):
    def __init__(self, num_spectrograms, num_spectrograms_per_epoch, seed=None) -> None:
        super().__init__(None)
        self._num_spectrograms = num_spectrograms
        self._num_spectrograms_per_epoch = num_spectrograms_per_epoch
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        return (
            self._rng.choice(self._num_spectrograms) for _ in range(self._num_spectrograms_per_epoch)
        )

    def __len__(self):
        return self._num_spectrograms_per_epoch


def get_voxceleb_task_dataloader(
    batch_size: int, 
    num_way: int, 
    num_support: int, 
    num_query: int, 
    num_tasks_per_epoch: int,
    voxceleb_dir: str,
    spectrograms_dir: str,
    dynamic_load: bool=True,
    resize: tuple=None,
    seed: int=None
):
    voxceleb_dir_path = Path(voxceleb_dir) / 'txt'
    spectrograms_dir_path = Path(spectrograms_dir)
    voxceleb_dataset = VoxCelebDataset(voxceleb_dir_path, spectrograms_dir_path, dynamic_load=dynamic_load, num_support=num_support, num_query=num_query, resize=resize, seed=seed) 
    return dataloader.DataLoader(
        dataset=voxceleb_dataset,
        batch_size=batch_size,
        sampler=VoxCelebTaskSampler(len(voxceleb_dataset.celebs), num_way, num_tasks_per_epoch),
        num_workers=8,
        collate_fn=lambda x: x,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


def get_voxceleb_dataloader(
    batch_size: int,
    num_iterations_per_epoch: int,
    voxceleb_dir: str,
    spectrograms_dir: str,
    dynamic_load: bool=True,
    resize: tuple=None,
    seed: int=None
):
    voxceleb_dir_path = Path(voxceleb_dir) / 'txt'
    spectrograms_dir_path = Path(spectrograms_dir)
    voxceleb_dataset = VoxCelebDataset(voxceleb_dir_path, spectrograms_dir_path, dynamic_load=dynamic_load, resize=resize, seed=seed) 
    return dataloader.DataLoader(
        dataset=voxceleb_dataset,
        batch_size=batch_size,
        sampler=VoxCelebSpectrogramSampler(len(voxceleb_dataset.spectrograms), num_iterations_per_epoch),
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
