import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import wavfile
from scipy import signal
import numpy as np
import torchaudio
import torch
import os
import shutil
from moviepy.editor import concatenate_videoclips, AudioFileClip, ImageClip
from moviepy.video.io.bindings import mplfig_to_npimage
import tempfile

from spectral_detector.spectrogram_tools import get_detections

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
teal_color = '#00FFFF'

def plot_spectrogram(paths, 
                  color='dusk', 
                  crop_time=None,
                  crop_frequency=None,  
                  resample_rate=48000, 
                  vmin=0, vmax=1,
                  show=False,
                  save_to=None,
                  return_vals=False,
                  vertical_line=None,
                  draw_boxes=False
                  ):
    if isinstance(paths, str) or not isinstance(paths, (list, tuple, np.ndarray)):
        paths = [paths]
    n_paths = len(paths)
    if isinstance(color, str) or not isinstance(color, (list, tuple, np.ndarray)):
        color = [color] * n_paths  # Extend color to match number of paths
    elif len(color) < n_paths:
        color = color + [color[-1]] * (n_paths - len(color))  # Fill up missing color
    if crop_time:
        if isinstance(crop_time[0], (list, tuple)) and len(crop_time) != n_paths:
            # If crop_time is not already adjusted to the number of paths
            crop_time = [crop_time[-1]] * n_paths
        elif isinstance(crop_time[0], (list, tuple)) and len(crop_time) == n_paths:
            # # Already in correct format
            pass
        else:
            # If crop_time is a single range, not wrapped within a list.
            crop_time = [crop_time] * n_paths
    if crop_frequency:
        if isinstance(crop_frequency[0], (list, tuple)) and len(crop_frequency) != n_paths:
            crop_frequency = [crop_frequency[-1]] * n_paths
        elif isinstance(crop_frequency[0], (list, tuple)) and len(crop_frequency) == n_paths:
            pass
        else:
            crop_frequency = [crop_frequency] * n_paths
    if vertical_line:
        if isinstance(vertical_line, (int, float)):
            vertical_line = [vertical_line] * n_paths
        elif len(vertical_line) < n_paths:
            vertical_line = vertical_line + [vertical_line[-1]] * (n_paths - len(vertical_line))

    n_fft = 2048
    hop_length=1024
    spec_transform = torchaudio.transforms.Spectrogram(#power spectrum
        n_fft=n_fft, 
        win_length=2048, 
        hop_length=hop_length, 
        power=2,
        window_fn=torch.hamming_window
    )

    specs = []
    base_waveform = None
    for i, file_path in enumerate(paths):
        # Read the audio file
        waveform, sample_rate = torchaudio.load(file_path, format='wav')
        resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
        waveform = resample(waveform)

        # Crop to timestamps
        if crop_time:
            start, end = crop_time[i]
            waveform = waveform[:, int(start*resample_rate):int(end*resample_rate)]
        if base_waveform is None:
            base_waveform = waveform

        spectrogram = spec_transform(waveform)
        spectrogram = 10 * torch.log10(spectrogram + 1e-10) # Convert to dB
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) # Normalize
        spectrogram = np.squeeze(spectrogram.numpy())
        if spectrogram.shape[0]==2: # mono
            spectrogram = np.mean(spectrogram, axis=0)

        times = np.linspace(0, spectrogram.shape[1] * hop_length / resample_rate, spectrogram.shape[1])
        frequencies = np.linspace(0, resample_rate / 2, spectrogram.shape[0])

        specs.append((spectrogram, times, frequencies))

    if draw_boxes:
        boxes = get_detections(paths, model_no=20)


        # # histogram plot
        # flat_spectrogram = spectrogram.flatten()
        # fig_hist, ax_hist = plt.subplots(figsize=(7.5, 4.5))
        # ax_hist.hist(flat_spectrogram, bins=100, density=True, alpha=0.75, color='b')
        # ax_hist.set_xlabel('Intensity (dB)')
        # ax_hist.set_ylabel('Count')

    if len(specs)==1:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        spectrogram, times, frequencies = specs[0]
        cax = ax.imshow(spectrogram, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]], vmin=vmin, vmax=vmax, cmap=custom_color_maps[color[0]])
        if vertical_line:
            ax.axvline(vertical_line[0], color='r')
        if draw_boxes:
            for box in boxes[0]:
                print(box) # xyxy normalized
                x1, y1, x2, y2 = box
                x1 = x1 * times[-1]
                x2 = x2 * times[-1]
                y1 = y1 * frequencies[-1]
                y2 = y2 * frequencies[-1]
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=teal_color, lw=1))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(cax, ax=ax, label='Intensity')

    elif len(specs)==2:
        fig = plt.figure(figsize=(7.5, 4.5))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # squished horizontal layout
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        axc = plt.subplot(gs[2])  # colorbar axis

        ax = [ax0, ax1]
        for i, (spectrogram, times, frequencies) in enumerate(specs):
            cax = ax[i].imshow(spectrogram, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]], vmin=vmin, vmax=vmax, cmap=custom_color_maps[color[i]])
            if vertical_line:
                ax[i].axvline(vertical_line[i], color='r')
            ax[i].set_xlabel('Time (s)')
            if i!=1: ax[i].set_ylabel('Frequency (Hz)')
            if i==1: fig.colorbar(cax, cax=axc, label='Intensity')

    elif len(specs)==3:
        fig = plt.figure(figsize=(7.5, 9))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05]) # vertical layout

        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1, 0])
        ax2 = plt.subplot(gs[2, 0])
        axc = plt.subplot(gs[:, 1])

        ax = [ax0, ax1, ax2]
        for i, (spectrogram, times, frequencies) in enumerate(specs):
            cax = ax[i].imshow(spectrogram, aspect='auto', origin='lower', extent=[times[0], times[-1], frequencies[0], frequencies[-1]], vmin=vmin, vmax=vmax, cmap=custom_color_maps[color[i]])
            if vertical_line:
                ax[i].axvline(vertical_line[i], color='r')
            if i==2: ax[i].set_xlabel('Time (s)')
            if i==1: ax[i].set_ylabel('Frequency (Hz)')
            if i==1: fig.colorbar(cax, cax=axc, label='Intensity')

    if show:
        plt.tight_layout()
        plt.show()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    if return_vals:
        return base_waveform, resample_rate

def generate_spectrogram_frames(audio_path, frames_folder='frames', frame_rate=30, crop_time=None):
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)
    
    base_waveform, sample_rate = plot_spectrogram(audio_path, crop_time=crop_time, return_vals=True)
    duration = base_waveform.shape[1] / sample_rate
    total_frames = int(np.ceil(frame_rate * duration))

    for i in range(total_frames):
        time_point = i / frame_rate
        frame_path = os.path.join(frames_folder, f"frame_{i:05}.png")
        plot_spectrogram(audio_path, 
                         crop_time=crop_time, 
                         vertical_line=time_point, 
                         save_to=frame_path)

    return total_frames, frame_rate, base_waveform, sample_rate

def create_video_with_audio(audio_paths, output_video_path="output_video.mp4", frame_rate=30, crop_time=None):
    frames_folder = "frames"
    total_frames, frame_rate, base_waveform, sample_rate = generate_spectrogram_frames(audio_paths, frames_folder, frame_rate, crop_time=crop_time)
    
    clips = [ImageClip(os.path.join(frames_folder, f"frame_{i:05}.png")).set_duration(1/frame_rate)
             for i in range(total_frames)]
    video_clip = concatenate_videoclips(clips, method="compose")

    with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as tmp_wav:
        torchaudio.save(tmp_wav.name, base_waveform, sample_rate)
        audio_clip = AudioFileClip(tmp_wav.name)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_video_path, fps=frame_rate)

file_path = "20230615_022500.WAV"
# paths = [file_path, file_path_2]
paths = [file_path]
# paths = [file_path, file_path_2, file_path_3]

output_video_path = "synced_video.mp4"
frame_rate = 30  # FPS for video
create_video_with_audio(paths, output_video_path, frame_rate, crop_time=[[36,45]])

# plot_spectrogram(paths, color='dusk', draw_boxes=False, show=True, crop_time=[[36,45]])