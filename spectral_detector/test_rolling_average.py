import torch
import torchaudio

file_path = "example/rāpaki_001_0224_15_06_2023_0207_16_06_2023/20230615_023900.WAV"

def rolling_avg_magnitude(file_path, window_length, frame_rate):
    """
    Compute the rolling average of an audio file for a given window length using PyTorch.

    :param file_path: Path to the audio file.
    :param window_length: Length of the rolling window in seconds.
    :param frame_rate: Frame rate of the audio file.
    :return: Tensor of rolling averages.
    """
    # Load the wave file
    waveform, sample_rate = torchaudio.load(file_path)

    # Ensure the waveform is mono (1D)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # Compute the number of samples per window
    samples_per_window = int(window_length * frame_rate)

    # Create a window tensor for convolution
    window = torch.ones(samples_per_window) / samples_per_window

    # Compute the rolling average using 1D convolution
    rolling_avg = torch.nn.functional.conv1d(torch.abs(waveform).unsqueeze(0), window.unsqueeze(0).unsqueeze(0), stride=samples_per_window).squeeze()

    return rolling_avg

frame_rate = 96000
rolling_avg_0 = rolling_avg_magnitude(file_path, 5, frame_rate)
rolling_avg_1 = rolling_avg_magnitude(file_path, 0.5, frame_rate)
rolling_avg_2 = rolling_avg_magnitude(file_path, 0.01, frame_rate)

# plot
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(3, 1, figsize=(5, 10))
axs[0].plot(np.arange(rolling_avg_0.shape[0]) / frame_rate, rolling_avg_0)
axs[0].set_title("10 sec rolling average")
axs[0].set_ylabel("Amplitude")
axs[1].plot(np.arange(rolling_avg_1.shape[0]) / frame_rate, rolling_avg_1)
axs[1].set_title("1 sec rolling average")
axs[1].set_ylabel("Amplitude")
axs[2].plot(np.arange(rolling_avg_2.shape[0]) / frame_rate, rolling_avg_2)
axs[2].set_title("0.05 sec rolling average")
axs[2].set_xlabel("Time (sec)")
axs[2].set_ylabel("Amplitude")
plt.tight_layout()
plt.show()
