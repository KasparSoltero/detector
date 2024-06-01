import os
import torchaudio
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib.ticker import PercentFormatter

# Paths for audio segments and noises
audio_segment_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/rap_001_isolated_manual'
audio_segment_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/manually_isolated_all'
audio_segment_tags_path = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/manually_isolated_all/tags.txt'
background_noise_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/noise'
other_noise_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/anth_and_wind'

# List of audio segment paths
audio_segment_paths = [os.path.join(audio_segment_root, f) for f in os.listdir(audio_segment_root) if f.endswith('.wav') or f.endswith('.WAV')]
background_noise_paths = [os.path.join(background_noise_root, f) for f in os.listdir(background_noise_root) if f.endswith('.wav') or f.endswith('.WAV')]
other_noise_paths = [os.path.join(other_noise_root, f) for f in os.listdir(other_noise_root) if f.endswith('.wav') or f.endswith('.WAV')]

# Function to load and transform audio into a spectrogram
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

def generate_overlays(n=5, sample_rate=48000, length=10, num_overlays=[0,5], num_other_sounds=[0,3], save_wav=False, plot=False, clear_dataset=False):
    # Loop for creating and overlaying spectrograms
    # DEFAULTS: 
        # noise normalised to 0.01 rms, 
        # song normalise to 0.01 * random power in range [0.4, 1.3]
        # songs can be cropped over edges, minimum 1 second present
        # images are normalised to 0.2 mean (manually set based on visual clarity)
        # 80:20 split train and val
        # 640x640 images
    
    if clear_dataset:
        os.system('rm -rf datasets/artificial_dataset/images/train/*')
        os.system('rm -rf datasets/artificial_dataset/images/val/*')
        os.system('rm -rf datasets/artificial_dataset/labels/train/*')
        os.system('rm -rf datasets/artificial_dataset/labels/val/*')
        os.system('rm -rf waveform_storage_mutable/*')

    val_index = int(n*0.8) # validation

    crop_len_seconds = length
    crop_len = crop_len_seconds * sample_rate  # 10 seconds at 48000 Hz

    # Initialize empty list to hold plot data
    spectrograms = []
    boxes_boxes = []
    boxes_boxes_2 = []
    # Loop to create and overlay audio
    for idx in range(n):
        specific_n_overlays = random.randint(num_overlays[0], num_overlays[1])
        print(f'{idx}:    creating image with {specific_n_overlays} vocalisations\n')
        # Select a random background noise (keep trying until one is long enough)
        bg_noise_audio = None
        while bg_noise_audio is None:
            bg_noise_path = random.choice(background_noise_paths)
            print(f'{idx}:    trying noisefile {os.path.basename(bg_noise_path)}\n')
            bg_noise_audio = load_spectrogram(bg_noise_path, crop_len=crop_len)
        
        # holds box cos
        boxes = []
        old_boxes = []
        # Adding random number of bird noises
        for j in range(specific_n_overlays):
            bird_noise_path = random.choice(audio_segment_paths)

            separation = None
            label = None
            if audio_segment_tags_path:
                with open(audio_segment_tags_path, 'r') as f:
                    for line in f:
                        specific_filename, specific_call_separation, specific_label = line.strip().split(', ')
                        if specific_filename == os.path.splitext(os.path.basename(bird_noise_path))[0]:
                            if not specific_call_separation=='x': separation = int(float(specific_call_separation) * (bg_noise_audio.shape[2]/crop_len_seconds))
                            if not specific_label=="x": label = specific_label
                            break
                    else:
                        print(f'error: no bird tag for {os.path.basename(bird_noise_path)}')

            print(f'{idx}:    {os.path.basename(bird_noise_path)} with separation {separation}\n')
            bird_noise_audio = load_spectrogram(bird_noise_path, power_range = [0.4,1.3])

            overlay = torch.zeros_like(bg_noise_audio)
            one_sec = int(bg_noise_audio.shape[2]/crop_len_seconds) #  1 second overlap

            minimum_start = min(0, one_sec-bird_noise_audio.shape[2])
            maximum_start = max(bg_noise_audio.shape[2]-bird_noise_audio.shape[2], bg_noise_audio.shape[2]-one_sec)
            start = random.randint(minimum_start, maximum_start)

            cropped_bird = bird_noise_audio[:,:, max(0,-start) : min(bg_noise_audio.shape[2]-start, bird_noise_audio.shape[2])]
            overlay[:,:,max(0,start) : max(0,start) + cropped_bird.shape[2]] = cropped_bird

            if separation:
                max_repetitions = int((bg_noise_audio.shape[2] - (start + bird_noise_audio.shape[2])) / (bird_noise_audio.shape[2] + separation))
                repetitions = random.randint(0, max_repetitions)
                print(f'{idx}:    {repetitions} repetitions of max {max_repetitions} possible\n')
                if repetitions: 
                    for i in range(repetitions):
                        overlay[:,:,start + (i+1)*(bird_noise_audio.shape[2] + separation) : start + (i+1)*(bird_noise_audio.shape[2] + separation) + bird_noise_audio.shape[2]] = bird_noise_audio

            bg_noise_audio += overlay

            temp_spec = np.squeeze(bird_noise_audio.numpy())

            # Initialize start and end offsets for time and frequency
            start_offset = 0
            end_offset = temp_spec.shape[1] - 1
            freq_start_offset = 0
            freq_end_offset = temp_spec.shape[0] - 1

            # Threshold value
            threshold = 1

            # Find time edges (horizontal)
            for i in range(temp_spec.shape[1]):
                if temp_spec[:, i].max() > threshold:
                    start_offset = i
                    break

            for i in range(temp_spec.shape[1] - 1, 0, -1):
                if temp_spec[:, i].max() > threshold:
                    end_offset = i
                    break

            # Find frequency edges (vertical) - minimum start at 2 (~100 Hz) to avoid low frequency interferance
            for i in range(2, temp_spec.shape[0]):
                if temp_spec[i, :].max() > threshold:
                    freq_start_offset = i
                    break

            for i in range(temp_spec.shape[0] - 1, 2, -1):
                if temp_spec[i, :].max() > threshold:
                    freq_end_offset = i
                    break

            boxes.append([max(0,start+start_offset), start+end_offset, freq_start_offset, freq_end_offset])
            old_boxes.append([max(0,start), start+temp_spec.shape[1]-1, 0, temp_spec.shape[0]-1])
            if separation:
                if repetitions:
                    for i in range(repetitions):
                        boxes.append([start+start_offset + (i+1)*(bird_noise_audio.shape[2] + separation), start+end_offset + (i+1)*(bird_noise_audio.shape[2] + separation), freq_start_offset, freq_end_offset])
                        old_boxes.append([start + (i+1)*(bird_noise_audio.shape[2] + separation), start + (i+1)*(bird_noise_audio.shape[2] + separation) + temp_spec.shape[1]-1, 0, temp_spec.shape[0]-1])

        # Adding random number of other noises from anth and wind
        specific_n_other_sounds = random.randint(num_other_sounds[0], num_other_sounds[1])
        for j in range(specific_n_other_sounds):
            other_noise_path = random.choice(other_noise_paths)
            other_noise_audio = load_spectrogram(other_noise_path, power_range = [0.1, 1.5])
            print(f'{idx}:other    {os.path.basename(other_noise_path)}\n')

            overlay = torch.zeros_like(bg_noise_audio)
            one_sec = int(bg_noise_audio.shape[2]/crop_len_seconds)

            minimum_start = min(0, one_sec-other_noise_audio.shape[2])
            maximum_start = max(bg_noise_audio.shape[2]-other_noise_audio.shape[2], bg_noise_audio.shape[2]-one_sec)
            start = random.randint(minimum_start, maximum_start)

            cropped_other = other_noise_audio[:,:, max(0,-start) : min(bg_noise_audio.shape[2]-start, other_noise_audio.shape[2])]
            overlay[:,:,max(0,start) : max(0,start) + cropped_other.shape[2]] = cropped_other
            bg_noise_audio += overlay

        # make ipeg image
        spec = np.squeeze(bg_noise_audio.numpy())
        # spec = (spec - np.min(spec)) / (np.ptp(spec)) #normalise to 0-1 # the max is typically an outlier so this normalisation is destructive
        spec = spec * 0.2 / spec.mean() # normalise to 0.2 mean
        spec = np.flipud(spec) #vertical flipping for image cos

        image_width, image_height = spec.shape[1], spec.shape[0]
        image = Image.fromarray(np.uint8(spec * 255), 'L')
        image = image.resize((640, 640), Image.Resampling.LANCZOS) #resize for yolo
        if idx>val_index:
            image.save(f'datasets/artificial_dataset/images/val/{idx}.jpg')
            txt_path = f'datasets/artificial_dataset/labels/val/{idx}.txt'
        else:
            image.save(f'datasets/artificial_dataset/images/train/{idx}.jpg')
            txt_path = f'datasets/artificial_dataset/labels/train/{idx}.txt'

        # make label txt file
        print(f'{idx}:    box locations: {boxes}')
        with open(txt_path, 'w') as f:
            for box in boxes:
                x_center = (box[0] + box[1]) / 2 / image_width
                width = (box[1] - box[0]) / image_width

                # vertical flipping for yolo
                y_center = (box[2] + box[3]) / 2 / image_height
                y_center = 1 - y_center
                height = (box[3] - box[2]) / image_height
                # Write to file in the format [class_id x_center y_center width height]
                f.write(f'0 {x_center} {y_center} {width} {height}\n')

        if(save_wav):   
            # convert back to waveform and save to wav for viewing testing
            waveform_transform = torchaudio.transforms.GriffinLim(
                n_fft=2048, 
                win_length=2048, 
                hop_length=512, 
                power=2.0
            )
            waveform = waveform_transform(bg_noise_audio)
            torchaudio.save(f"waveform_storage_mutable/{idx}.wav", waveform, sample_rate=sample_rate)

        # for plots 
        boxes_boxes.append(boxes)
        spectrograms.append(np.squeeze(bg_noise_audio.numpy()))
        boxes_boxes_2.append(old_boxes)

    if(plot):
        # Plotting the spectrograms
        fig, axes = plt.subplots(n, 1, figsize=(10, 10))
        if n == 1:
            axes = [axes]
        for ax, spec, boxes, boxes_2 in zip(axes, spectrograms, boxes_boxes, boxes_boxes_2):
            cax = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax)
            
            for box in boxes:
                print(box)
                color = 'red'
                start_x, end_x, start_y, end_y = box
                ax.axhline(end_y, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linewidth=1)
                ax.axhline(start_y, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linewidth=1)
                ax.axvline(start_x, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linewidth=1)
                ax.axvline(end_x, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linewidth=1)
            for box in boxes_2:
                print(box)
                color = 'green'
                start_x, end_x, start_y, end_y = box
                ax.axhline(end_y, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linestyle='--', linewidth=1)
                ax.axhline(start_y, xmin=start_x/image_width, xmax=end_x/image_width, color=color, linestyle='--', linewidth=1)
                ax.axvline(start_x, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linestyle='--', linewidth=1)
                ax.axvline(end_x, ymin=start_y/image_height, ymax=end_y/image_height, color=color, linestyle='--', linewidth=1)

            ax.set_xlim(0, image_width)
            ax.set_ylim(0, image_height)

        plt.tight_layout()
        plt.show()

    return spectrograms, boxes_boxes

generate_overlays(
    n=5000,
    sample_rate=48000,
    length=10,
    num_overlays=[0,5],
    num_other_sounds=[0,2],
    save_wav=False, 
    plot=False,
    clear_dataset=True
    )