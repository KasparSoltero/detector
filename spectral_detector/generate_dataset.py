import os
import torchaudio
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import csv
import chardet

from matplotlib.ticker import PercentFormatter

def read_tags(path):
    # reads a csv, returns dictionaries of filenames with each column's attributes
    if os.path.exists(path):
        with open(path, 'rb') as raw_file:
            result = chardet.detect(raw_file.read())
            encoding = result['encoding']
        
        with open(path, mode='r', newline='', encoding=encoding) as file:
            reader = csv.DictReader(file)
            tags_data = {}
            for row in reader:
                filename = row['filename']
                tags_data[filename] = {}
                for header in reader.fieldnames:
                    tags_data[filename][header] = row[header]
                
        return tags_data
    else:
        raise FileNotFoundError(f'No tags file found for {path}')

# Paths for audio segments and noises
# audio_segment_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/rap_001_isolated_manual'
# audio_segment_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/manually_isolated_all'
# audio_segment_tags_path = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/manually_isolated_all/tags.txt'
# background_noise_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/noise'
# other_noise_root = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/anth_and_wind'

data_root = '/Users/kaspar/Documents/ecoacoustics/data/manually_isolated'
background_path = 'background_noise'
positive_paths = ['unknown', 'amphibian', 'reptile', 'mammal', 'insect', 'bird']
negative_paths = ['anthrophony', 'geophony']

positive_segment_paths = []
positive_datatags = {}
for path in positive_paths:
    tags_data = read_tags(os.path.join(data_root, path, 'tags.csv'))
    positive_datatags[path] = tags_data
    for f in os.listdir(os.path.join(data_root, path)):
        if f.endswith('.wav') or f.endswith('.WAV'):
            if not f.startswith('atemporal'):
                #  overlay_label for tracing training data, e.g. 5th bird -> bi5
                positive_datatags[path][f]['overlay_label'] = path[:2]+str(list(tags_data.keys()).index(f))
                audio_segment_path = os.path.join(data_root, path, f)
                positive_segment_paths.append(audio_segment_path)
            else:
                print(f'TODO handle atemporal files: {f} ({path})')

negative_segment_paths = []
negative_datatags = {}
for path in negative_paths:
    tags_data = read_tags(os.path.join(data_root, path, 'tags.csv'))
    negative_datatags[path] = tags_data
    for f in os.listdir(os.path.join(data_root, path)):
        if f.endswith('.wav') or f.endswith('.WAV'):
            negative_datatags[path][f]['overlay_label'] = path[:2]+str(list(tags_data.keys()).index(f))
            negative_segment_paths.append(os.path.join(data_root, path, f))

background_noise_paths = []
background_datatags = read_tags(os.path.join(data_root, background_path, 'tags.csv'))
for f in os.listdir(os.path.join(data_root, background_path)):
    if f.endswith('.wav') or f.endswith('.WAV'):
        background_datatags[f]['overlay_label'] = 'bg'+str(list(background_datatags.keys()).index(f))
        background_noise_paths.append(os.path.join(data_root, background_path, f))

# Scatter plot coordinates
# # plot coordinates
# coordinates = []
# years = []
# for path in positive_datatags:
#     for filename in positive_datatags[path]:
#         if 'overlay_label' in positive_datatags[path][filename]:
#             latitude = float(positive_datatags[path][filename]['lat'])
#             longitude = float(positive_datatags[path][filename]['long'])
#             coordinates.append((latitude, longitude))
#             years.append(int(positive_datatags[path][filename]['date']))
# y_coords, x_coords = zip(*coordinates)
# plt.scatter(x_coords, y_coords, c=years, cmap='viridis', alpha=0.8, marker='o')
# coordinates = []
# years = []
# for filename in background_datatags:
#     if 'overlay_label' in background_datatags[filename]:
#         latitude = float(background_datatags[filename]['lat'])
#         longitude = float(background_datatags[filename]['long'])
#         coordinates.append((latitude, longitude))
#         years.append(int(background_datatags[filename]['date']))
# y_coords2, x_coords2 = zip(*coordinates)
# plt.scatter(x_coords2, y_coords2, c=years, cmap='viridis', alpha=0.8, marker='x')
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.colorbar(label="Year")
# plt.grid(True)
# plt.show()

# # List of audio segment paths
# positive_segment_paths = [os.path.join(audio_segment_root, f) for f in os.listdir(audio_segment_root) if f.endswith('.wav') or f.endswith('.WAV')]
# background_noise_paths = [os.path.join(background_noise_root, f) for f in os.listdir(background_noise_root) if f.endswith('.wav') or f.endswith('.WAV')]
# negative_segment_paths = [os.path.join(other_noise_root, f) for f in os.listdir(other_noise_root) if f.endswith('.wav') or f.endswith('.WAV')]

# Function to load and transform audio into a spectrogram
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
        # TODO power should be dB? or sqaures? 100x distance range -> max*(1-100)^-2
        random_power = random.uniform(power_range[0], power_range[1])
        spec *= random_power
        return spec, random_power
    return spec

def generate_overlays(
        n=5, 
        sample_rate=48000, 
        final_length_seconds=10, 
        positive_overlay_range=[0,5], 
        num_other_sounds=[0,3], 
        save_wav=False, 
        plot=False, 
        clear_dataset=False
    ):
    # Loop for creating and overlaying spectrograms
    # DEFAULTS: 
        # noise normalised to 1 rms, dB
        # song normalise to 1 rms, dB, * random power in range [0.5, 1.5]
        # song bbox threshold 30 dB (1000)
        # songs can be cropped over edges, minimum 1 second present
        # images are normalised to 0-100 dB, then 0-1 to 255
        # 80:20 split train and val
        # 640x640 images
    
    if clear_dataset:
        os.system('rm -rf datasets/artificial_dataset/images/train/*')
        os.system('rm -rf datasets/artificial_dataset/images/val/*')
        os.system('rm -rf datasets/artificial_dataset/labels/train/*')
        os.system('rm -rf datasets/artificial_dataset/labels/val/*')
        os.system('rm -rf waveform_storage_mutable/*')

    val_index = int(n*0.8) # validation

    # Initialize empty list to hold plot data
    plots_spectrograms = []
    plots_boxes = []

    # Loop to create and overlay audio
    for idx in range(n):
        label = str(idx) # image label
        boxes = [] # holds box cos

        # Select a random background noise (keep trying until one is long enough)
        bg_noise_audio = None
        while bg_noise_audio is None:
            bg_noise_path = random.choice(background_noise_paths)
            bg_noise_audio = load_spectrogram(bg_noise_path, crop_len=final_length_seconds * sample_rate)
        label += '_' + background_datatags[os.path.basename(bg_noise_path)]['overlay_label']

        final_freq_bins, final_time_bins = bg_noise_audio.shape[1], bg_noise_audio.shape[2]
        one_sec_bins = int(final_time_bins/final_length_seconds) #  1 second overlap in samples

        # highpass filter set by background noise
        highpass_hz = int(background_datatags[os.path.basename(bg_noise_path)]['highpass'])
        freq_bins_cutoff = int(highpass_hz / (sample_rate / 2) * final_freq_bins)

        # Adding random number of positive vocalisation noises
        n_positive_overlays = random.randint(positive_overlay_range[0], positive_overlay_range[1])
        print(f'\n{idx}:    creating new image with {n_positive_overlays} positive overlays, bg={os.path.basename(bg_noise_path)}')
        for j in range(n_positive_overlays):
            # select positive overlay
            positive_segment_path = random.choice(positive_segment_paths)
            
            ## TODO fix separation
            # separation = None
            # label = None
            # if audio_segment_tags_path:
            #     with open(audio_segment_tags_path, 'r') as f:
            #         for line in f:
            #             specific_filename, specific_call_separation, specific_label = line.strip().split(', ')
            #             if specific_filename == os.path.splitext(os.path.basename(positive_segment_path))[0]:
            #                 if not specific_call_separation=='x': separation = int(float(specific_call_separation) * (bg_noise_audio.shape[2]/final_length_seconds))
            #                 if not specific_label=="x": label = specific_label
            #                 break
            #         else:
            #             print(f'error: no bird tag for {os.path.basename(positive_segment_path)}')

            # load positive overlay
            # TODO verify power overlay
            positive_segment, power = load_spectrogram(positive_segment_path, power_range = [0.2,1.2])
            print(f'{idx}:    adding +{power:.2f},  {os.path.basename(positive_segment_path)}')
            
            # overlay the positive segment on the background noise
            overlay = torch.zeros_like(bg_noise_audio)
            
            # calculate position of segment in background noise
            seg_freq_bins, seg_time_bins = positive_segment.shape[1], positive_segment.shape[2]

            # attempt to place segment at least 0.5 seconds from other starts
            minimum_start = min(0, one_sec_bins-seg_time_bins)
            maximum_start = max(final_time_bins-seg_time_bins, final_time_bins-one_sec_bins)
            for i in range(20):
                start_time = random.randint(minimum_start, maximum_start)
                if not any([start_time < box[0] + 0.5*one_sec_bins and start_time > box[0] - 0.5*one_sec_bins for box in boxes]):
                    break

            # dynamically find bounding box based on power
            threshold = 5 # dB difference between maximum power in segment and mean power in background noise at the segment location
            
            # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
            freq_start = 0
            freq_end = seg_freq_bins - 1
            for i in range(max(2, freq_bins_cutoff), seg_freq_bins):
                noise_power_at_freq_slice = bg_noise_audio[:, i, max(start_time,0):min(start_time+seg_time_bins,final_time_bins-1)].mean()
                noise_power_at_freq_slice = max(0,10 * torch.log10(noise_power_at_freq_slice + 1e-6))
                positive_segment_at_freq_slice = 10 * torch.log10(positive_segment[:, i, :].max() + 1e-6)
                if (positive_segment_at_freq_slice - noise_power_at_freq_slice) > threshold:
                    freq_start = i
                    break
            for i in range(seg_freq_bins - 1, max(2, freq_bins_cutoff), -1):
                noise_power_at_freq_slice = bg_noise_audio[:, i, max(start_time,0):min(start_time+seg_time_bins,final_time_bins-1)].mean()
                noise_power_at_freq_slice = max(0,10 * torch.log10(noise_power_at_freq_slice + 1e-6))
                positive_segment_at_freq_slice = 10 * torch.log10(positive_segment[:, i, :].max() + 1e-6)
                if (positive_segment_at_freq_slice - noise_power_at_freq_slice) > threshold:
                    freq_end = i
                    break
            
            start_time_offset = 0
            end_time_offset = (seg_time_bins - 1) - max(0, start_time + seg_time_bins - final_time_bins)
            # Find time edges (horizontal scan)
            for i in range(seg_time_bins):
                noise_power_at_time_slice = bg_noise_audio[:, freq_start:freq_end, min(max(start_time+i,0),final_time_bins-1)].mean()
                noise_power_at_time_slice = max(0,10 * torch.log10(noise_power_at_time_slice + 1e-6))
                positive_segment_at_time_slice = 10 * torch.log10(positive_segment[:, :, i].max() + 1e-6)
                if (positive_segment_at_time_slice - noise_power_at_time_slice) > threshold:
                    start_time_offset = i
                    break
            for i in range((seg_time_bins - 1) - max(0, start_time + seg_time_bins - final_time_bins), 0, -1):
                noise_power_at_time_slice = bg_noise_audio[:, freq_start:freq_end, min(max(start_time+i,0),final_time_bins-1)].mean()
                noise_power_at_time_slice = max(0,10 * torch.log10(noise_power_at_time_slice + 1e-6))
                positive_segment_at_time_slice = 10 * torch.log10(positive_segment[:, :, i].max() + 1e-6)
                if (positive_segment_at_time_slice - noise_power_at_time_slice) > threshold:
                    end_time_offset = i
                    break
            
            # check height and width are not less than 1% of the final image
            if ((freq_end - freq_start)/final_freq_bins) < 0.0065 or ((end_time_offset - start_time_offset)/final_time_bins) < 0.0065:
                print(f"{idx}: Error, too small, power {power:.3f}, freq {(freq_end - freq_start)/final_freq_bins:.3f}, time {(end_time_offset - start_time_offset)/final_time_bins:.3f}")
                continue
            if ((freq_end - freq_start)/final_freq_bins) > 0.99:
                print(f"{idx}: Error, too faint, power {power:.3f}")
                continue

            positive_segment_cropped = positive_segment[:,:, max(0,-start_time) : min(final_time_bins-start_time, seg_time_bins)]
            overlay[:,:,max(0,start_time) : max(0,start_time) + positive_segment_cropped.shape[2]] = positive_segment_cropped

            #TODO separation
            # if separation:
            #     max_repetitions = int((bg_noise_audio.shape[2] - (start + positive_segment.shape[2])) / (positive_segment.shape[2] + separation))
            #     repetitions = random.randint(0, max_repetitions)
            #     print(f'{idx}:    {repetitions} repetitions of max {max_repetitions} possible\n')
            #     if repetitions: 
            #         for i in range(repetitions):
            #             overlay[:,:,start + (i+1)*(positive_segment.shape[2] + separation) : start + (i+1)*(positive_segment.shape[2] + separation) + positive_segment.shape[2]] = positive_segment

            bg_noise_audio += overlay

            # add bounding box to list, in units of spectrogram time and frequency bins
            boxes.append([max(0,start_time+start_time_offset), start_time+end_time_offset, freq_start, freq_end])
            label += '_' + positive_datatags[os.path.basename(os.path.dirname(positive_segment_path))][os.path.basename(positive_segment_path)]['overlay_label']
            label += 'p'+str(power)[:3] # power label
            # old_boxes.append([max(0,start), start+pos_seg_2d.shape[1]-1, 0, pos_seg_2d.shape[0]-1])
            # if separation:
            #     if repetitions:
            #         for i in range(repetitions):
            #             boxes.append([start+start_offset + (i+1)*(positive_segment.shape[2] + separation), start+end_offset + (i+1)*(positive_segment.shape[2] + separation), freq_start_offset, freq_end_offset])
            #             old_boxes.append([start + (i+1)*(positive_segment.shape[2] + separation), start + (i+1)*(positive_segment.shape[2] + separation) + pos_seg_2d.shape[1]-1, 0, pos_seg_2d.shape[0]-1])

        # adding random number of negative noises (cars, rain, wind). no boxes stored for these
        n_negative_overlays = random.randint(num_other_sounds[0], num_other_sounds[1])
        for j in range(n_negative_overlays):
            negative_segment_path = random.choice(negative_segment_paths)
            label += '_'+negative_datatags[os.path.basename(os.path.dirname(negative_segment_path))][os.path.basename(negative_segment_path)]['overlay_label']
            negative_segment, power = load_spectrogram(negative_segment_path, power_range = [0.5, 1.5])
            label += 'p'+str(power)[:3] # power label
            print(f'{idx}:    neg {power:.2f},  ({os.path.basename(negative_segment_path)})')
            
            overlay = torch.zeros_like(bg_noise_audio)
            one_sec_samp = int(bg_noise_audio.shape[2]/final_length_seconds)

            minimum_start = min(0, one_sec_samp-negative_segment.shape[2])
            maximum_start = max(bg_noise_audio.shape[2]-negative_segment.shape[2], bg_noise_audio.shape[2]-one_sec_samp)
            start = random.randint(minimum_start, maximum_start)

            negative_segment_cropped = negative_segment[:,:, max(0,-start) : min(bg_noise_audio.shape[2]-start, negative_segment.shape[2])]
            overlay[:,:,max(0,start) : max(0,start) + negative_segment_cropped.shape[2]] = negative_segment_cropped
            bg_noise_audio += overlay

        # apply highpass filter, convert to dB
        bg_noise_audio[:, :freq_bins_cutoff, :] = 0
        bg_noise_audio = 10 * torch.log10(bg_noise_audio + 1e-6) # to dB
        # normalise to 0-1
        spec_clamped = bg_noise_audio.clamp(min=0) #clamp bottom to 0 dB
        spec_normalised = spec_clamped / spec_clamped.max()

        # make jpeg image
        spec_2d = np.squeeze(spec_normalised.numpy())
        spec = np.flipud(spec_2d) #vertical flipping for image cos

        value = spec # intensity of the spectrogram -> value in hsv
        saturation = 4 * value * (1 - value) # parabola for saturation
        hue = np.linspace(0,1,final_freq_bins)[:, np.newaxis] # linearly spaced hues over frequency range
        hue = np.tile(hue, (1, final_time_bins))
        
        hsv_spec = np.stack([hue, saturation, value], axis=-1)
        rgb_spec = hsv_to_rgb(hsv_spec) # convert to rgb
        # image = Image.fromarray(np.uint8(spec * 255), 'L') # greyscale
        image = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
        image = image.resize((640, 640), Image.Resampling.LANCZOS) #resize for yolo

        if idx > val_index:
            image_output_path = f'datasets/artificial_dataset/images/val/{label}.jpg'
            txt_output_path = f'datasets/artificial_dataset/labels/val/{label}.txt'
        else:
            image_output_path = f'datasets/artificial_dataset/images/train/{label}.jpg'
            txt_output_path = f'datasets/artificial_dataset/labels/train/{label}.txt'

        image.save(image_output_path, format='JPEG', quality=95)
        # try:
        #     img = Image.open(image_output_path)
        #     img.load()  # loading of image data
        #     img.close()
        # except (IOError, SyntaxError) as e:
        #     print(f"Invalid image after reopening: {e}")

        # make label txt file
        print(f'{idx}:    {label},    box locations: {boxes}\n')
        with open(txt_output_path, 'w') as f:
            for box in boxes:
                x_center = (box[0] + box[1]) / 2 / final_time_bins
                width = (box[1] - box[0]) / final_time_bins

                # vertical flipping for yolo
                y_center = (box[2] + box[3]) / 2 / final_freq_bins
                y_center = 1 - y_center
                height = (box[3] - box[2]) / final_freq_bins

                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                    print(f"{idx}: Error, box out of bounds!\n\n******\n\n******\n\n*******\n\n")

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
            # invert db
            bg_noise_audio_off_db = 10 ** (bg_noise_audio / 10)
            waveform = waveform_transform(bg_noise_audio_off_db)
            rms = torch.sqrt(torch.mean(torch.square(waveform)))
            waveform = waveform * 0.1/rms
            torchaudio.save(f"waveform_storage_mutable/{label}.wav", waveform, sample_rate=sample_rate)

        # for plots 
        plots_boxes.append(boxes)
        plots_spectrograms.append(np.squeeze(spec_normalised.numpy()))

    if(plot):
        # Plotting the spectrograms
        # Calculate the number of rows needed
        rows = (n // 3) + (1 if n % 3 != 0 else 0)

        # Plotting the spectrograms
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))

        # Flatten the axes for easy iteration
        axes = axes.flatten()

        for i, (spec, boxes) in enumerate(zip(plots_spectrograms, plots_boxes)):
            ax = axes[i]
            im = ax.imshow(spec, aspect='equal', origin='lower', cmap='viridis', vmin=0, vmax=1)
            
            for box in boxes:
                color = 'red'
                start_x, end_x, start_y, end_y = box
                ax.axhline(end_y, xmin=start_x / final_time_bins, xmax=end_x / final_time_bins, color=color, linewidth=1)
                ax.axhline(start_y, xmin=start_x / final_time_bins, xmax=end_x / final_time_bins, color=color, linewidth=1)
                ax.axvline(start_x, ymin=start_y / final_freq_bins, ymax=end_y / final_freq_bins, color=color, linewidth=1)
                ax.axvline(end_x, ymin=start_y / final_freq_bins, ymax=end_y / final_freq_bins, color=color, linewidth=1)

            ax.set_xlim(0, final_time_bins)
            ax.set_ylim(0, final_freq_bins)
            ax.set_aspect(aspect='equal')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    return plots_spectrograms, plots_boxes

generate_overlays(
    n=5000,
    sample_rate=48000,
    final_length_seconds=10,
    positive_overlay_range=[0,5],
    num_other_sounds=[0,0],
    save_wav=False, 
    plot=False,
    clear_dataset=True
    )