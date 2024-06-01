import glob
import os
import random
import torchaudio
import torch
import matplotlib.pyplot as plt
import numpy as np

directory = '/Users/kaspar/Documents/AEDI/data/manually_isolated_all'

def load_spectrogram(path, crop_len=None, normalised=True, power_range=None):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, 48000)
    waveform = resample(waveform)

    if waveform.shape[0]>1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # mono
    if crop_len:
        if waveform.shape[1] > crop_len:
            start = random.randint(0, waveform.shape[1] - crop_len)
            waveform = waveform[:, start:start+crop_len]
        else:
            print("Error: Background noise is shorter than 10 seconds.")
            return None
    if normalised:
        print(f'\r\r normalised {os.path.basename(path)} to 0.01 rms')
        rms = torch.sqrt(torch.mean(waveform ** 2))
        waveform = waveform*0.01/rms
    if power_range:
        random_power = random.uniform(power_range[0], power_range[1])
        waveform = waveform * random_power

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    return spec_transform(waveform)

for file_path in glob.glob(directory + '/*.WAV'):
    spec = np.squeeze(load_spectrogram(file_path).numpy())
    spec = spec * 0.2 / spec.mean()

    # find the first and last point where the max is above 0.05
    start = 0
    for i in range(spec.shape[1]):
        if spec[:, i].max() > 0.5:
            start = i
            break
    end = spec.shape[1] - 1
    for i in range(spec.shape[1] - 1, 0, -1):
        if spec[:, i].max() > 0.5:
            end = i
            break

    fig, axs = plt.subplots(3)
    fig.suptitle(os.path.basename(file_path))
    axs[0].imshow(spec, aspect='auto', origin='lower', vmin=0, vmax=1)
    axs[0].axvline(start, color='r', linestyle='--')
    axs[0].axvline(end, color='r', linestyle='--')
    axs[1].plot(spec.mean(axis=0))
    axs[2].plot(spec.max(axis=0))

    # plt.imshow(spec, aspect='auto', origin='lower', vmin=0, vmax=1)
    plt.show()

# directory = 'datasets/artificial_dataset/labels/train'

# negative_labels = []

# for file_path in glob.glob(directory + '/*.txt'):
#     with open(file_path, 'r') as file:
#         labels = file.readlines()
#         for label in labels:
#             values = label.split()
#             if float(values[3]) < 0:
#                 negative_labels.append(file_path)
#                 print(values[3])
#                 break

# if negative_labels:
#     print("The following files contain labels with negative values:")
#     for file_path in negative_labels:
#         print(file_path)
# else:
#     print("No files contain labels with negative values.")
