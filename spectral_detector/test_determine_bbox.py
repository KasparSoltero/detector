## determine the bounding box of a vocalisation
import os
import random
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

dusk_colors = [
    (255, 255, 255),
(255, 253, 255),
(255, 251, 255),
(255, 249, 255),
(255, 247, 255),
(255, 245, 255),
(255, 243, 255),
(255, 241, 255),
(255, 239, 255),
(255, 237, 255),
(255, 235, 255),
(255, 233, 255),
(255, 231, 255),
(255, 229, 255),
(255, 227, 255),
(255, 225, 255),
(255, 223, 255),
(255, 221, 255),
(255, 219, 255),
(255, 217, 255),
(255, 215, 255),
(255, 213, 255),
(255, 211, 255),
(255, 209, 255),
(255, 207, 255),
(255, 205, 255),
(255, 203, 255),
(255, 201, 255),
(255, 199, 255),
(255, 197, 255),
(255, 195, 255),
(255, 193, 255),
(255, 191, 255),
(255, 189, 255),
(255, 187, 255),
(255, 185, 255),
(255, 183, 255),
(255, 181, 255),
(255, 179, 255),
(255, 177, 255),
(255, 175, 255),
(255, 173, 255),
(255, 172, 255),
(255, 170, 255),
(255, 168, 255),
(255, 166, 255),
(255, 164, 255),
(255, 162, 255),
(255, 160, 255),
(255, 158, 255),
(255, 156, 255),
(255, 154, 255),
(255, 152, 255),
(255, 150, 255),
(255, 148, 255),
(255, 146, 255),
(255, 144, 255),
(255, 142, 255),
(255, 140, 255),
(255, 138, 255),
(255, 136, 255),
(255, 134, 255),
(255, 132, 255),
(255, 130, 255),
(255, 128, 255),
(253, 126, 255),
(251, 124, 255),
(249, 122, 255),
(247, 120, 255),
(245, 118, 255),
(242, 116, 255),
(241, 114, 255),
(238, 112, 255),
(237, 110, 255),
(235, 108, 255),
(233, 106, 255),
(231, 104, 255),
(229, 102, 255),
(227, 100, 255),
(225, 98, 255),
(223, 96, 255),
(221, 94, 255),
(219, 92, 255),
(217, 90, 255),
(215, 88, 255),
(213, 86, 255),
(211, 84, 255),
(209, 81, 255),
(207, 79, 255),
(205, 77, 255),
(203, 75, 255),
(201, 73, 255),
(199, 71, 255),
(197, 69, 255),
(195, 67, 255),
(193, 65, 255),
(191, 63, 255),
(189, 61, 255),
(187, 59, 255),
(185, 57, 255),
(183, 55, 255),
(181, 53, 255),
(179, 51, 255),
(177, 49, 255),
(175, 47, 255),
(173, 45, 255),
(171, 43, 255),
(169, 41, 255),
(167, 39, 255),
(165, 37, 255),
(163, 35, 255),
(161, 33, 255),
(159, 31, 255),
(157, 29, 255),
(155, 27, 255),
(153, 25, 255),
(151, 23, 255),
(149, 21, 255),
(147, 19, 255),
(145, 17, 255),
(143, 15, 255),
(141, 13, 255),
(138, 11, 255),
(136, 9, 255),
(134, 7, 255),
(132, 5, 255),
(131, 3, 255),
(129, 1, 255),
(126, 0, 254),
(125, 0, 252),
(122, 0, 250),
(121, 0, 248),
(118, 0, 246),
(116, 0, 244),
(115, 0, 242),
(113, 0, 240),
(111, 0, 238),
(109, 0, 236),
(107, 0, 234),
(105, 0, 232),
(102, 0, 230),
(100, 0, 228),
(98, 0, 227),
(97, 0, 225),
(94, 0, 223),
(93, 0, 221),
(91, 0, 219),
(89, 0, 217),
(87, 0, 215),
(84, 0, 213),
(83, 0, 211),
(81, 0, 209),
(79, 0, 207),
(77, 0, 205),
(75, 0, 203),
(73, 0, 201),
(70, 0, 199),
(68, 0, 197),
(66, 0, 195),
(64, 0, 193),
(63, 0, 191),
(61, 0, 189),
(59, 0, 187),
(57, 0, 185),
(54, 0, 183),
(52, 0, 181),
(51, 0, 179),
(49, 0, 177),
(47, 0, 175),
(44, 0, 174),
(42, 0, 172),
(40, 0, 170),
(39, 0, 168),
(37, 0, 166),
(34, 0, 164),
(33, 0, 162),
(31, 0, 160),
(29, 0, 158),
(27, 0, 156),
(25, 0, 154),
(22, 0, 152),
(20, 0, 150),
(18, 0, 148),
(17, 0, 146),
(14, 0, 144),
(13, 0, 142),
(11, 0, 140),
(9, 0, 138),
(6, 0, 136),
(4, 0, 134),
(2, 0, 132),
(0, 0, 130),
(0, 0, 128),
(0, 0, 126),
(0, 0, 124),
(0, 0, 122),
(0, 0, 120),
(0, 0, 118),
(0, 0, 116),
(0, 0, 114),
(0, 0, 112),
(0, 0, 110),
(0, 0, 108),
(0, 0, 106),
(0, 0, 104),
(0, 0, 102),
(0, 0, 100),
(0, 0, 98),
(0, 0, 96),
(0, 0, 94),
(0, 0, 92),
(0, 0, 90),
(0, 0, 88),
(0, 0, 86),
(0, 0, 83),
(0, 0, 81),
(0, 0, 79),
(0, 0, 77),
(0, 0, 75),
(0, 0, 73),
(0, 0, 71),
(0, 0, 69),
(0, 0, 67),
(0, 0, 65),
(0, 0, 63),
(0, 0, 61),
(0, 0, 59),
(0, 0, 57),
(0, 0, 55),
(0, 0, 53),
(0, 0, 51),
(0, 0, 49),
(0, 0, 47),
(0, 0, 45),
(0, 0, 43),
(0, 0, 41),
(0, 0, 39),
(0, 0, 37),
(0, 0, 35),
(0, 0, 33),
(0, 0, 31),
(0, 0, 29),
(0, 0, 26),
(0, 0, 24),
(0, 0, 22),
(0, 0, 20),
(0, 0, 18),
(0, 0, 16),
(0, 0, 14),
(0, 0, 12),
(0, 0, 10),
(0, 0, 8),
(0, 0, 6),
(0, 0, 4),
(0, 0, 2),
(0, 0, 0),
]
dusk_colors.reverse()
dusk_colormap = LinearSegmentedColormap.from_list("dusk", [tuple(color/255 for color in c) for c in dusk_colors])
custom_color_maps = {
    'dusk': dusk_colormap
}

# Duplicate Function to load and transform audio into a spectrogram
def load_spectrogram(path, crop_len=None, normalised=True, power_range=None):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, 48000)
    waveform = resample(waveform)

    if waveform.shape[0]>1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # mono
    if crop_len: #random crop
        if waveform.shape[1] > crop_len:
            start = random.randint(0, waveform.shape[1] - crop_len)
            waveform = waveform[:, start:start+crop_len]
        else:
            print(f"Error: {path} is shorter than crop length")
            return None
    if normalised:
        rms = torch.sqrt(torch.mean(torch.square(waveform)))
        waveform = waveform*1/rms

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0,
        window_fn=torch.hamming_window
    )
    spec = spec_transform(waveform)
    
    if power_range:
        random_power = random.uniform(power_range[0], power_range[1])
        spec *= random_power
        return spec, random_power
    return spec

data_root = '/Users/kaspar/Documents/ecoacoustics/data/manually_isolated'
background_path = 'bg_temp'
positive_paths = ['unknown', 'amphibian', 'reptile', 'mammal', 'insect', 'bird']
negative_paths = ['anthrophony', 'geophony']

audio_segment_paths = []
for path in positive_paths:
    for f in os.listdir(os.path.join(data_root, path)):
        if f.endswith('.wav') or f.endswith('.WAV'):
            if not f.startswith('atemporal'):
                audio_segment_path = os.path.join(data_root, path, f)
                audio_segment_paths.append(audio_segment_path)
            else:
                print(f'TODO handle atemporal files: {f} ({path})')

background_noise_paths = []
for f in os.listdir(os.path.join(data_root, background_path)):
    if f.endswith('.wav') or f.endswith('.WAV'):
        background_noise_paths.append(os.path.join(data_root, background_path, f))

# thresholds = [0.1,0.2,0.5,1,2,10]
# threshold_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

thresholds = [1,15,25]
# thresholds = [inv_db(t) for t in thresholds]
print(thresholds)
threshold_colors = ['green', '#00FFFF', 'red']
n = 4
final_length_seconds = 3
sample_rate = 48000

boxes = []
specs = []
for i in range(n):

    # Select a random background noise (keep trying until one is long enough)
    bg_noise_audio = None
    while bg_noise_audio is None:
        bg_noise_path = random.choice(background_noise_paths)
        bg_noise_audio = load_spectrogram(bg_noise_path, crop_len=final_length_seconds * sample_rate)
    final_freq_bins, final_time_bins = bg_noise_audio.shape[1], bg_noise_audio.shape[2]
    one_sec_bins = int(final_time_bins/final_length_seconds) #  1 second overlap in samples

    positive_segment_path = random.choice(audio_segment_paths)
    positive_segment, positive_segment_power = load_spectrogram(positive_segment_path, power_range = [0.8,2])
    print(positive_segment_power)

    # spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) # Normalize
    spec_2d = np.squeeze(positive_segment.numpy())
    freq_bins, time_bins = spec_2d.shape
    freq_bins_cutoff = 0

    # calculate position of segment in background noise
    minimum_start = min(0, one_sec_bins-time_bins)
    maximum_start = max(final_time_bins-time_bins, final_time_bins-one_sec_bins)
    start_time = random.randint(minimum_start, maximum_start)

    boxes_n = []
    for threshold in thresholds:
        # Initialize start and end offsets for time and frequency
        # initialize start and end frequency offsets
        freq_start = 0
        freq_end = freq_bins - 1

        # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
        for i in range(max(2, freq_bins_cutoff), freq_bins):
            noise_power_at_freq_slice = bg_noise_audio[:, i, max(start_time,0):min(start_time+time_bins,final_time_bins-1)].mean()
            noise_power_at_freq_slice = max(0,10 * torch.log10(noise_power_at_freq_slice + 1e-6))
            positive_segment_at_freq_slice = 10 * torch.log10(positive_segment[:, i, :].max() + 1e-6)
            if (positive_segment_at_freq_slice - noise_power_at_freq_slice) > threshold:
                freq_start = i
                break
        for i in range(freq_bins - 1, max(2, freq_bins_cutoff), -1):
            noise_power_at_freq_slice = bg_noise_audio[:, i, max(start_time,0):min(start_time+time_bins,final_time_bins-1)].mean()
            noise_power_at_freq_slice = max(0,10 * torch.log10(noise_power_at_freq_slice + 1e-6))
            positive_segment_at_freq_slice = 10 * torch.log10(positive_segment[:, i, :].max() + 1e-6)
            if (positive_segment_at_freq_slice - noise_power_at_freq_slice) > threshold:
                freq_end = i
                break
        
        start_time_offset = 0
        end_time_offset = time_bins - 1
        # Find time edges (horizontal scan)
        for i in range(time_bins):
            noise_power_at_time_slice = bg_noise_audio[:, freq_start:freq_end, min(max(start_time+i,0),final_time_bins-1)].mean()
            noise_power_at_time_slice = max(0,10 * torch.log10(noise_power_at_time_slice + 1e-6))
            positive_segment_at_time_slice = 10 * torch.log10(positive_segment[:, :, i].max() + 1e-6)
            if (positive_segment_at_time_slice - noise_power_at_time_slice) > threshold:
                start_time_offset = i
                break
        for i in range(time_bins - 1, 0, -1):
            noise_power_at_time_slice = bg_noise_audio[:, freq_start:freq_end, min(max(start_time+i,0),final_time_bins-1)].mean()
            noise_power_at_time_slice = max(0,10 * torch.log10(noise_power_at_time_slice + 1e-6))
            positive_segment_at_time_slice = 10 * torch.log10(positive_segment[:, :, i].max() + 1e-6)
            if (positive_segment_at_time_slice - noise_power_at_time_slice) > threshold:
                end_time_offset = i
                break

        boxes_n.append([max(0,start_time+start_time_offset), min(start_time+end_time_offset,final_time_bins), freq_start, freq_end])

    # overlay the positive segment on the background noise
    overlay = torch.zeros_like(bg_noise_audio)
    positive_segment_cropped = positive_segment[:,:, max(0,-start_time) : min(final_time_bins-start_time, time_bins)]
    overlay[:,:,max(0,start_time) : max(0,start_time) + positive_segment_cropped.shape[2]] = positive_segment_cropped

    bg_noise_audio += overlay


    hop_length=512
    resample_rate = 48000
    times = np.linspace(0, bg_noise_audio.shape[2] * hop_length / resample_rate, bg_noise_audio.shape[2])
    frequencies = np.linspace(0, resample_rate / 2, bg_noise_audio.shape[1])
    specs.append((np.squeeze(bg_noise_audio.numpy()), times, frequencies))

    boxes.append(boxes_n)

# Plotting the spectrograms
fig = plt.figure(figsize=(7.5, 9))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])  # squished horizontal layout
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[3])
ax3 = plt.subplot(gs[4])
axc = plt.subplot(gs[:,2])
ax = [ax0, ax1, ax2, ax3]

for i, (spectrogram, times, frequencies) in enumerate(specs):

    # to dB (numpy)
    spectrogram = 10 * np.log10(spectrogram + 1e-6)    
    # normalise to 0-1
    spec_clamped = np.clip(spectrogram, 0, None) # clamp bottom to 0 dB
    spectrogram = spec_clamped / spec_clamped.max()

    cax = ax[i].imshow(spectrogram, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]], vmin=0, vmax=1, cmap=custom_color_maps['dusk'])
    image_width, image_height = spectrogram.shape[1], spectrogram.shape[0]
    # image_width = times[-1]
    # image_height = frequencies[-1]
    
    for box, color in zip(boxes[i], threshold_colors):
        start_x, end_x, start_y, end_y = box
        ax[i].axhline(frequencies[-1]*end_y/image_height, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linewidth=1, linestyle='--')
        ax[i].axhline(frequencies[-1]*start_y/image_height, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linewidth=1, linestyle='--')
        ax[i].axvline(times[-1]*start_x/image_width, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linewidth=1, linestyle='--')
        ax[i].axvline(times[-1]*end_x/image_width, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linewidth=1, linestyle='--')
    
    if i==2 or i==3:
        ax[i].set_xlabel('Time (s)')
    if i==0 or i==2: 
        ax[i].set_ylabel('Frequency (Hz)')
    if i==2:
        fig.colorbar(cax, cax=axc, label='Intensity')
    if i==1 or i==3:
        ax[i].set_yticks([])

plt.tight_layout()
plt.show()