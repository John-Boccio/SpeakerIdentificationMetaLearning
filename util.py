import torch
import matplotlib.pyplot as plt
import numpy as np


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()


def plot_spectrogram(spectrogram):
    freq = np.arange(256)
    time = np.arange(301) * 0.01
    plt.pcolormesh(time, freq, torch.flip(spectrogram, dims=(0,)))
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')
    plt.show()


def plot_reconstruction(spectrogram, mask, reconstruction, mask_method, mask_ratio, title=None):
    # util.plot_reconstruction(image_batch[0,0].detach().cpu(), mask[0, 0].detach().cpu(), x[0, 0].detach().cpu(), self._mask, self._mask_ratio)
    fig = plt.figure()
    axd = fig.subplot_mosaic([
        ['spectrogram', 'masked_spectrogram'],
        ['missing_spectrogram', 'missing_spectrogram_reconstructed']
    ])

    freq = np.arange(256)
    time = np.arange(301) * 0.01
    axd['spectrogram'].pcolormesh(time, freq, torch.flip(spectrogram, dims=(0,)))
    axd['spectrogram'].set_xlabel('Time (s)')
    axd['spectrogram'].set_ylabel('Frequency Bin')
    axd['spectrogram'].set_title('Spectrogram')

    axd['masked_spectrogram'].pcolormesh(time, freq, torch.flip(spectrogram * mask, dims=(0,)))
    axd['masked_spectrogram'].set_xlabel('Time (s)')
    axd['masked_spectrogram'].set_ylabel('Frequency Bin')
    axd['masked_spectrogram'].set_title('Masked Spectrogram')

    axd['missing_spectrogram'].pcolormesh(time, freq, torch.flip(spectrogram * (-mask + 1), dims=(0,)))
    axd['missing_spectrogram'].set_xlabel('Time (s)')
    axd['missing_spectrogram'].set_ylabel('Frequency Bin')
    axd['missing_spectrogram'].set_title('Missing Spectrogram')

    axd['missing_spectrogram_reconstructed'].pcolormesh(time, freq, torch.flip(reconstruction * (-mask + 1), dims=(0,)))
    axd['missing_spectrogram_reconstructed'].set_xlabel('Time (s)')
    axd['missing_spectrogram_reconstructed'].set_ylabel('Frequency Bin')
    axd['missing_spectrogram_reconstructed'].set_title('Missing Spectrogram Reconstructed')

    if title is None:
        title = f'Masked Auto-Encoder Spectrogram Reconstruction (Mask Method, Ratio = {mask_method}, {mask_ratio})'
    fig_title = fig.suptitle(title)
    fig_title.set_fontsize(20)
    fig.show()
