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
from spectrogram_tools import spectrogram_transformed, load_spectrogram
from matplotlib.ticker import PercentFormatter

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

def plot_labels(idx=[0,-1], save_directory='datasets_mutable'):
    # Plotting the labelsq
    # Plotting the spectrograms
    # Calculate the number of rows needed
    if idx[1] == -1:
        idx[1] = len(os.listdir(f'{save_directory}/artificial_dataset/images/train'))
    rows = (idx[1] - idx[0]) //  3 + 1 if (idx[1] - idx[0]) else 1
    if rows > 4:
        rows = 4
    # Plotting the spectrograms
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))

    # Ensure axes is a 2D array
    if rows == 1:
        axes = [axes]  # In case of a single row, make it a list of lists

    for i, image_path in enumerate(os.listdir(f'{save_directory}/artificial_dataset/images/train')[idx[0]:idx[1]]):
        # Compute row and column index
        row_idx = i // 3
        col_idx = i % 3
        if row_idx >= rows:
            break
        
        image = Image.open(f'{save_directory}/artificial_dataset/images/train/{image_path}')
        label_path = f'{save_directory}/artificial_dataset/labels/train/{image_path[:-4]}.txt'
        # get the corresponding label
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())
                boxes.append([class_id, x_center, y_center, width, height])

        # plot image
        ax = axes[row_idx][col_idx]
        ax.imshow(np.array(image))

        # plot boxes
        for box in boxes:
            x_center, y_center, width, height = box[1:]
            x_min = (x_center - width / 2) * image.width
            y_min = (y_center - height / 2) * image.height
            if box[0] == 0:
                rect = plt.Rectangle((x_min, y_min), width * image.width, height * image.height, linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
            elif box[0] == 1:
                rect = plt.Rectangle((x_min, y_min), width * image.width, height * image.height, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
            ax.add_patch(rect)

        ax.set_title(f'{image_path}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def save_spectrogram(spec, path):
    path = path+'_spectrogram.jpg'
    image = spectrogram_transformed(spec, to_pil=True, resize=(640, 640))
    image.save(path, format='JPEG', quality=95)

def spec_to_audio(spec, energy_type='power', save_to=None, normalise_rms=0.02):
    # convert back to waveform and save to wav for viewing testing
    waveform_transform = torchaudio.transforms.GriffinLim(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    # if energy_type=='dB':
    #     normalise = 'dB_to_power'
    # elif energy_type=='power':
    #     normalise = None
    # spec_audio = spectrogram_transformed(spec, normalise=normalise, to_torch=True)
    # if energy_type=='complex':
    spec_audio = torch.square(torch.abs(spec))
    waveform = waveform_transform(spec_audio)
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    waveform = waveform*normalise_rms/rms
    if save_to:
        torchaudio.save(f"{save_to}.wav", waveform, sample_rate=48000)

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
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename = filename[:-4]
                tags_data[filename] = {}
                for header in reader.fieldnames:
                    tags_data[filename][header] = row[header]
                
        return tags_data
    else:
        raise FileNotFoundError(f'No tags file found for {path}')

def load_input_dataset(data_root, background_path, positive_paths, negative_paths):
    positive_segment_paths = []
    positive_datatags = {}
    for path in positive_paths:
        tags_data = read_tags(os.path.join(data_root, path, 'tags.csv'))
        positive_datatags[path] = tags_data
        for f in os.listdir(os.path.join(data_root, path)):
            if f.endswith('.wav') or f.endswith('.WAV'):
                found_filename = f[:-4]
                # TODO handle atemporal files
                #  overlay_label for tracing training data, e.g. 5th bird -> bi5
                positive_datatags[path][found_filename]['overlay_label'] = path[:2]+str(list(tags_data.keys()).index(found_filename))
                full_audio_path = os.path.join(data_root, path, f)
                positive_segment_paths.append(full_audio_path)

    negative_segment_paths = []
    negative_datatags = {}
    for path in negative_paths:
        tags_data = read_tags(os.path.join(data_root, path, 'tags.csv'))
        negative_datatags[path] = tags_data
        for f in os.listdir(os.path.join(data_root, path)):
            if f.endswith('.wav') or f.endswith('.WAV'):
                found_filename = f[:-4]
                negative_datatags[path][found_filename]['overlay_label'] = path[:2]+str(list(tags_data.keys()).index(found_filename))
                negative_segment_paths.append(os.path.join(data_root, path, f))

    background_noise_paths = []
    background_datatags = read_tags(os.path.join(data_root, background_path, 'tags.csv'))
    for f in os.listdir(os.path.join(data_root, background_path)):
        if f.endswith('.wav') or f.endswith('.WAV'):
            found_filename = f[:-4]
            background_datatags[found_filename]['overlay_label'] = 'bg'+str(list(background_datatags.keys()).index(found_filename))
            background_noise_paths.append(os.path.join(data_root, background_path, f))

    return positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags

def generate_overlays(
        get_data_paths=[None, None, None, None],
        save_directory='datasets_mutable',
        n=1,
        sample_rate=48000,
        final_length_seconds=10,
        positive_overlay_range=[1,1],
        negative_overlay_range=[0,0],
        save_wav=False,
        plot=False,
        clear_dataset=False,
        val_ratio = 0.8,
        noise_power_range=[10, 60],
        signal_power_range=[5,20]
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
        os.system(f'rm -rf {save_directory}/artificial_dataset/images/train/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/images/val/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/labels/train/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/labels/val/*')
        os.system(f'rm -rf {save_directory}/waveform_storage_mutable/*')

    data_root, background_path, positive_paths, negative_paths = get_data_paths
    if data_root is None:
        data_root='../data/manually_isolated'
        background_path='background_noise'
    if positive_paths is None:
        positive_paths = ['unknown', 'amphibian', 'reptile', 'mammal', 'insect', 'bird']
        negative_paths = ['anthrophony', 'geophony']
    positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags = load_input_dataset(data_root, background_path, positive_paths, negative_paths)

    val_index = int(n*val_ratio) # validation
    # Loop to create and overlay audio
    for idx in range(n):
        label = str(idx) # image label
        # Select a random background noise (keep trying until one is long enough)
        bg_noise_audio = None
        while bg_noise_audio is None:
            bg_noise_path = random.choice(background_noise_paths)
            bg_noise_audio = load_spectrogram(bg_noise_path, random_crop=final_length_seconds, unit_type='complex')
        # bg_noise_audio = spectrogram_transformed(bg_noise_audio, set_rms=1)
        bg_noise_audio, bg_noise_power = spectrogram_transformed(
            bg_noise_audio, 
            set_snr_db=random.uniform(noise_power_range[0], noise_power_range[1])
        )
        label += '_' + background_datatags[os.path.basename(bg_noise_path)[:-4]]['overlay_label']

        final_freq_bins, final_time_bins = bg_noise_audio.shape[1], bg_noise_audio.shape[2]
        one_sec_bins = int(final_time_bins/final_length_seconds) #  1 second overlap in samples
        # highpass filter set by background noise tags data
        highpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]]['highpass']
        if highpass_hz:
            highpass_hz = int(highpass_hz)
        else:
            highpass_hz = 0
        highpass_hz += random.randint(0,100)
        lowpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]]['lowpass']
        if lowpass_hz:
            lowpass_hz = int(lowpass_hz)
        else:
            lowpass_hz = sample_rate / 2
        lowpass_hz -= random.randint(0,1000)
        freq_bins_cutoff_bottom = int(highpass_hz / (sample_rate / 2) * final_freq_bins)
        freq_bins_cutoff_top = int(lowpass_hz / (sample_rate / 2) * final_freq_bins)

        # adding random number of negative noises (cars, rain, wind). 
        # no boxes stored for these, as they are treated like background noise
        n_negative_overlays = random.randint(negative_overlay_range[0], negative_overlay_range[1])
        for j in range(n_negative_overlays):
            negative_segment_path = random.choice(negative_segment_paths)
            label += '_'+negative_datatags[os.path.basename(os.path.dirname(negative_segment_path))][os.path.basename(negative_segment_path)[:-4]]['overlay_label']
            
            negative_segment = load_spectrogram(negative_segment_path, unit_type='complex')
            
            overlay = torch.zeros_like(bg_noise_audio)
            one_sec_samp = int(bg_noise_audio.shape[2]/final_length_seconds)

            if negative_segment.shape[2] > final_time_bins:
                minimum_start = final_time_bins - negative_segment.shape[2]
                maximum_start = 0
            else:
                minimum_start = min(0, one_sec_samp-negative_segment.shape[2])
                maximum_start = max(bg_noise_audio.shape[2]-negative_segment.shape[2], bg_noise_audio.shape[2]-one_sec_samp)
            start = random.randint(minimum_start, maximum_start)

            negative_segment_cropped = negative_segment[:,:, max(0,-start) : min(bg_noise_audio.shape[2]-start, negative_segment.shape[2])]
            
            # normalised_segment = spectrogram_transformed(negative_segment_cropped, set_rms=1)
            normalised_segment, negative_power = spectrogram_transformed(
                negative_segment_cropped, 
                set_snr_db=random.uniform(signal_power_range[0], signal_power_range[1])
            )
            label += 'p'+str(negative_power/bg_noise_power)[:3] # power label
            print(f'{idx}:    neg {negative_power/bg_noise_power:.2f},  ({os.path.basename(negative_segment_path)})')
            
            overlay[:,:,max(0,start) : max(0,start) + normalised_segment.shape[2]] = normalised_segment
            bg_noise_audio += overlay

        # copy
        final_audio = bg_noise_audio.clone()
        # Adding random number of positive vocalisation noises
        # initialise boxes
        boxes = [] # holds box cos
        classes = []
        n_positive_overlays = random.randint(positive_overlay_range[0], positive_overlay_range[1])
        print(f'\n{idx}:    creating new image with {n_positive_overlays} positive overlays, bg={os.path.basename(bg_noise_path)}')
        for j in range(n_positive_overlays):
            # select positive overlay
            positive_segment_path = random.choice(positive_segment_paths)
            
            ## TODO fix separation
            
            # positive_segment, power = load_spectrogram(positive_segment_path, power_range = [1,30])
            positive_segment = load_spectrogram(positive_segment_path, unit_type='complex')
            # check if 'species' is 'chorus'
            if positive_datatags[os.path.basename(os.path.dirname(positive_segment_path))][os.path.basename(positive_segment_path)[:-4]]['species'] == 'chorus':
                if classes.count(1) > 0:
                    continue # only one chorus per image
                species_class=1
            else:
                species_class=0
            
            # calculate position of segment in background noise
            seg_freq_bins, seg_time_bins = positive_segment.shape[1], positive_segment.shape[2]

            def calculate_db_power_(spec, start, end, freq_start, freq_end, threshold=0, type='mean'):
                if type=='mean':
                    cropped = spec[:, freq_start:freq_end, start:end]
                    return 10 * torch.log10(cropped[cropped > threshold].mean() + 1e-6)                  
                elif type=='max':
                    return 10 * torch.log10(spec[:, freq_start:freq_end, start:end].max() + 1e-6)

            def calculate_power(spec, start, end, freq_start, freq_end, threshold=0, type='mean'):
                # Calculate the magnitude of the complex spectrogram
                magnitude = torch.square(torch.abs(spec))

                if type == 'mean':
                    # Crop the specified range
                    cropped = magnitude[:, freq_start:freq_end, start:end]

                    # Apply threshold and calculate the mean
                    mean_val = cropped[cropped > threshold].mean()

                    # Convert mean to decibels
                    return mean_val
                
                elif type == 'max':
                    # Crop the specified range
                    cropped = magnitude[:, freq_start:freq_end, start:end]
                    
                    # Find the max value
                    max_val = cropped.max()
                    
                    # Convert max to decibels
                    return max_val

            # dynamically find bounding box based on power
            threshold = 1 # snr
            found=0
            if seg_time_bins < final_time_bins:
                # attempt to place segment at least 0.5 seconds from other starts
                minimum_start = min(0, one_sec_bins-seg_time_bins)
                maximum_start = max(final_time_bins-seg_time_bins, final_time_bins-one_sec_bins)
                for i in range(20):
                    start_time = random.randint(minimum_start, maximum_start)
                    if not any([start_time < box[0] + 0.5*one_sec_bins and start_time > box[0] - 0.5*one_sec_bins for box in boxes]):
                        break
                        
                positive_segment = positive_segment[:,:, max(0,-start_time) : min(final_time_bins-start_time, seg_time_bins)]
                normalised_segment, power = spectrogram_transformed(
                    positive_segment, 
                    set_snr_db=random.uniform(signal_power_range[0], signal_power_range[1]),
                )
                cropped_time_bins = normalised_segment.shape[2]
                
                # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
                freq_start = 0
                freq_end = seg_freq_bins - 1
                for i in range(2, seg_freq_bins-1):
                    noise_power_at_freq_slice = calculate_power(
                        bg_noise_audio, 
                        max(start_time,0), 
                        min(start_time+cropped_time_bins,final_time_bins-1), 
                        min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top)+1)
                    positive_segment_at_freq_slice = calculate_power(
                        normalised_segment, 
                        0, cropped_time_bins, 
                        i, i+1,
                        type='max')
                    if 10*torch.log10(positive_segment_at_freq_slice / noise_power_at_freq_slice) > threshold:
                        freq_start = i
                        found+=1
                        break
                for i in range(seg_freq_bins - 1, 2, -1):
                    noise_power_at_freq_slice = calculate_power(
                        bg_noise_audio, 
                        max(start_time,0), min(start_time+cropped_time_bins,final_time_bins-1), 
                        min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top)+1)
                    positive_segment_at_freq_slice = calculate_power(
                        normalised_segment,
                        0,cropped_time_bins, 
                        i, i+1,
                        type='max')
                    if 10*torch.log10(positive_segment_at_freq_slice / noise_power_at_freq_slice) > threshold:
                        freq_end = i
                        found+=1
                        break
            
                # Find time edges (horizontal scan)
                start_time_offset = 0
                end_time_offset = (cropped_time_bins - 1)
                for i in range(cropped_time_bins-1):
                    noise_power_at_time_slice = calculate_power(
                        bg_noise_audio, 
                        min(max(start_time+i,0),final_time_bins-1), min(max(start_time+i,0),final_time_bins-1)+1, 
                        min(max(freq_start,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(freq_end,freq_bins_cutoff_bottom),freq_bins_cutoff_top),
                        type='mean')
                    positive_segment_at_time_slice = calculate_power(
                        normalised_segment, 
                        i,i+1, 
                        freq_start, freq_end, type='max')
                    if 10*torch.log10(positive_segment_at_time_slice / noise_power_at_time_slice) > threshold:
                        start_time_offset = i
                        found+=1
                        break
                for i in range((cropped_time_bins-1), 0, -1):
                    noise_power_at_time_slice = calculate_power(
                        bg_noise_audio, 
                        min(max(start_time+i,0),final_time_bins-1), min(max(start_time+i,0),final_time_bins-1)+1, 
                        min(max(freq_start,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(freq_end,freq_bins_cutoff_bottom),freq_bins_cutoff_top),
                        type='mean')
                    positive_segment_at_time_slice = calculate_power(
                        normalised_segment, 
                        i,i+1, 
                        freq_start, freq_end, type='max')
                    if 10*torch.log10(positive_segment_at_time_slice / noise_power_at_time_slice) > threshold:
                        end_time_offset = i
                        found+=1
                        break
            # noises longer than final length are treated as continuous, aren't cropped over edges
            elif seg_time_bins >= (final_time_bins*0.95):
                # #TODO remove
                ok = ['r001_20230615_194000_01.WAV']
                if (species_class==0 and not (os.path.basename(positive_segment_path) in ok)):
                    raise ValueError(f"{idx}:{os.path.basename(positive_segment_path)} is longer than final length and not chorus, skipping")
                
                minimum_start = final_time_bins - seg_time_bins
                maximum_start = 0
                start_time = random.randint(minimum_start, maximum_start)
                
                positive_segment_cropped = positive_segment[:,:, max(0,-start_time) : min(final_time_bins-start_time, seg_time_bins)]
                normalised_segment, power = spectrogram_transformed(
                    positive_segment_cropped,
                    set_snr_db=random.uniform(signal_power_range[0], signal_power_range[1])
                )
                cropped_time_bins = normalised_segment.shape[2]

                # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
                freq_start = 0
                freq_end = seg_freq_bins - 1
                for i in range(2, seg_freq_bins-1):
                    noise_power_at_freq_slice = calculate_power(
                        bg_noise_audio, 
                        max(start_time,0), 
                        min(start_time+cropped_time_bins,final_time_bins-1), 
                        min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top)+1)
                    positive_segment_at_freq_slice = calculate_power(
                        normalised_segment, 
                        0,cropped_time_bins, 
                        i, i+1, threshold=threshold,type='max')
                    if 10*torch.log10(positive_segment_at_freq_slice / noise_power_at_freq_slice) > threshold:
                        freq_start = i
                        found+=1
                        break
                for i in range(seg_freq_bins - 1, 2, -1):
                    noise_power_at_freq_slice = calculate_power(
                        bg_noise_audio, 
                        max(start_time,0),min(start_time+cropped_time_bins,final_time_bins-1), 
                        min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top), min(max(i,freq_bins_cutoff_bottom),freq_bins_cutoff_top)+1)
                    positive_segment_at_freq_slice = calculate_power(
                        normalised_segment, 
                        0, cropped_time_bins, 
                        i, i+1, threshold=threshold,type='max')
                    if 10*torch.log10(positive_segment_at_freq_slice / noise_power_at_freq_slice) > threshold:
                        freq_end = i
                        found+=1
                        break

                start_time_offset = 0
                end_time_offset = cropped_time_bins-1
                found+=2

            freq_start = max(min(freq_start, freq_bins_cutoff_top), freq_bins_cutoff_bottom)
            freq_end = min(max(freq_end, freq_bins_cutoff_bottom), freq_bins_cutoff_top)

            # check height and width are not less than 1% of the final image
            if ((freq_end - freq_start)/final_freq_bins) < 0.0065 or ((end_time_offset - start_time_offset)/final_time_bins) < 0.0065:
                print(f"{idx}: Error, too small, power {power/bg_noise_power:.3f}, freq {(freq_end - freq_start)/final_freq_bins:.3f}, time {(end_time_offset - start_time_offset)/final_time_bins:.3f}")
                continue
            if ((freq_end - freq_start)/final_freq_bins) > 0.99 or found < 4:
                print(f"{idx}: Error, too faint, power {power/bg_noise_power:.3f}")
                continue

            print(f'power: {power:.2f}, bg_power: {bg_noise_power:.2f}')

            normalised_segment = spectrogram_transformed(
                normalised_segment,
                highpass=freq_bins_cutoff_bottom,
                lowpass=freq_bins_cutoff_top
            )
            # overlay the positive segment on the background noise
            overlay = torch.zeros_like(bg_noise_audio)
            overlay[:,:,max(0,start_time) : max(0,start_time) + normalised_segment.shape[2]] = normalised_segment

            final_audio += overlay

            # add bounding box to list, in units of spectrogram time and frequency bins
            boxes.append([max(start_time_offset,start_time+start_time_offset), max(end_time_offset, start_time+end_time_offset), freq_start, freq_end])
            classes.append(species_class)
            label += positive_datatags[os.path.basename(os.path.dirname(positive_segment_path))][os.path.basename(positive_segment_path)[:-4]]['overlay_label']
            label += 'p'+str(power/bg_noise_power)[:4] # power label
            # old_boxes.append([max(0,start), start+pos_seg_2d.shape[1]-1, 0, pos_seg_2d.shape[0]-1])
            # if separation:
            #     if repetitions:
            #         for i in range(repetitions):
            #             boxes.append([start+start_offset + (i+1)*(positive_segment.shape[2] + separation), start+end_offset + (i+1)*(positive_segment.shape[2] + separation), freq_start_offset, freq_end_offset])
            #             old_boxes.append([start + (i+1)*(positive_segment.shape[2] + separation), start + (i+1)*(positive_segment.shape[2] + separation) + pos_seg_2d.shape[1]-1, 0, pos_seg_2d.shape[0]-1])

        final_audio = spectrogram_transformed(
            final_audio,
            highpass=freq_bins_cutoff_bottom,
            lowpass=freq_bins_cutoff_top)

        if(save_wav):
            wav_path = f"{save_directory}/waveform_storage_mutable/{label}"
            spec_to_audio(final_audio, save_to=wav_path, energy_type='complex')
        # final_audio = spectrogram_transformed(
        #     final_audio, 
        #     set_rms=1
        # )
        image = spectrogram_transformed(
            final_audio,
            to_pil=True,
            normalise='complex_to_PCEN',
            resize=(640, 640),
        )
        if idx > val_index:
            image_output_path = f'{save_directory}/artificial_dataset/images/val/{label}.jpg'
            txt_output_path = f'{save_directory}/artificial_dataset/labels/val/{label}.txt'
        else:
            image_output_path = f'{save_directory}/artificial_dataset/images/train/{label}.jpg'
            txt_output_path = f'{save_directory}/artificial_dataset/labels/train/{label}.txt'
        
        if len(image_output_path) > 100:
            image_output_path = image_output_path[:90]+'.jpg'
            txt_output_path = txt_output_path[:90]+'.txt'
        
        image.save(image_output_path, format='JPEG', quality=95)
        # try:
        #     img = Image.open(image_output_path)
        #     img.load()  # loading of image data
        #     img.close()
        # except (IOError, SyntaxError) as e:
        #     print(f"Invalid image after reopening: {e}")

        def calculate_iou(box1, box2):
            # Coordinates of the intersection rectangle
            x_left = max(box1[0], box2[0])
            y_top = max(box1[2], box2[2])
            x_right = min(box1[1], box2[1])
            y_bottom = min(box1[3], box2[3])

            # Calculate intersection area
            if x_right < x_left or y_bottom < y_top:
                return 0.0  # No intersection
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # Area of both the bounding boxes
            box1_area = (box1[1] - box1[0]) * (box1[3] - box1[2])
            box2_area = (box2[1] - box2[0]) * (box2[3] - box2[2])

            # intersection_over_smaller = intersection_area / min(box1_area, box2_area)
            intersection_over_box2 = intersection_area / box2_area
            # Calculate IoU
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            return intersection_over_box2

        def combine_boxes(box1, box2):
            x_min = min(box1[0], box2[0])
            x_max = max(box1[1], box2[1])
            y_min = min(box1[2], box2[2])
            y_max = max(box1[3], box2[3])
            return [x_min, x_max, y_min, y_max]

        def find(parent, i):
            if parent[i] == i:
                return i
            parent[i] = find(parent, parent[i])  # Path compression
            return parent[i]

        def union(parent, rank, i, j):
            root_i = find(parent, i)
            root_j = find(parent, j)
            
            if root_i != root_j:
                if rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                elif rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1

        def merge_boxes(boxes, classes, iou_threshold=0.5):
            merged_boxes = []
            merged_classes = []
            print(f'species: {classes}')
            print(f'boxes: {boxes}')
            boxes_to_merge = []
            for i, (box, species_class) in enumerate(zip(boxes, classes)):
                for k in range(len(boxes) - 1, -1, -1):
                    if k == i:
                        continue
                    print(f'comparing box {i} to {k}. species: {species_class} vs {classes[k]}')
                    if species_class != classes[i]:
                        continue
                    other_box = boxes[k]
                    if calculate_iou(box, other_box) > iou_threshold:
                        print('merging')
                        boxes_to_merge.append([i,k])

            # # Combine all boxes to merge
            # if boxes_to_merge:
                

            return merged_boxes, merged_classes

        def merge_boxes_by_class(boxes, classes, iou_threshold=0.5):
            parent = list(range(len(boxes)))
            rank = [0] * len(boxes)
            
            for i, (box, species_class) in enumerate(zip(boxes, classes)):
                for k in range(len(boxes) - 1, -1, -1):
                    if k == i:
                        continue
                    if species_class != classes[k]:
                        continue
                    other_box = boxes[k]
                    if calculate_iou(box, other_box) > iou_threshold:
                        union(parent, rank, i, k)
            
            merged_boxes = {}
            for i in range(len(boxes)):
                root = find(parent, i)
                if root not in merged_boxes:
                    merged_boxes[root] = boxes[i]
                else:
                    merged_boxes[root] = combine_boxes(merged_boxes[root], boxes[i])
            
            # Propagate the merge to ensure indirect overlaps are included
            updated = True
            while updated:
                updated = False
                temp_merged_boxes = merged_boxes.copy()
                for root1 in list(temp_merged_boxes.keys()):
                    for root2 in list(temp_merged_boxes.keys()):
                        if root1 == root2:
                            continue
                        if classes[root1] != classes[root2]:  # Ensure same class
                            continue
                        if calculate_iou(temp_merged_boxes[root1], temp_merged_boxes[root2]) > iou_threshold:
                            temp_merged_boxes[root1] = combine_boxes(temp_merged_boxes[root1], temp_merged_boxes[root2])
                            del temp_merged_boxes[root2]
                            updated = True
                            break
                    if updated:
                        break
                merged_boxes = temp_merged_boxes

            # Extract the final merged boxes and their corresponding classes
            final_boxes = list(merged_boxes.values())
            final_classes = []
            for root in merged_boxes:
                final_classes.append(classes[root])
            
            return final_boxes, final_classes

        # Merge boxes based on IoU
        merged_boxes, merged_classes = merge_boxes_by_class(boxes, classes, iou_threshold=0.5)
        # make label txt file
        print(f'{idx}:    {label},    box locations: {merged_boxes}\n')
        with open(txt_output_path, 'w') as f:
            for box, species_class in zip(merged_boxes, merged_classes):
                x_center = (box[0] + box[1]) / 2 / final_time_bins
                width = (box[1] - box[0]) / final_time_bins

                # vertical flipping for yolo
                y_center = (box[2] + box[3]) / 2 / final_freq_bins
                y_center = 1 - y_center
                height = (box[3] - box[2]) / final_freq_bins

                if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                    print(f"{idx}: Error, box out of bounds!\n\n******\n\n******\n\n*******\n\n")

                # Write to file in the format [class_id x_center y_center width height]
                f.write(f'{species_class} {x_center} {y_center} {width} {height}\n')

    if(plot):
        plot_labels([0,n], save_directory)

        # # Plotting the spectrograms
        # # Calculate the number of rows needed
        # rows = (n // 3) + (1 if n % 3 != 0 else 0)

        # # Plotting the spectrograms
        # fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))

        # # Flatten the axes for easy iteration
        # axes = axes.flatten()

        # for i, (spec, boxes) in enumerate(zip(plots_spectrograms, plots_boxes)):
        #     ax = axes[i]
        #     im = ax.imshow(spec, aspect='equal', origin='lower', cmap='viridis', vmin=0, vmax=1)
            
        #     for box in boxes:
        #         color = 'red'
        #         start_x, end_x, start_y, end_y = box
        #         ax.axhline(end_y, xmin=start_x / final_time_bins, xmax=end_x / final_time_bins, color=color, linewidth=1)
        #         ax.axhline(start_y, xmin=start_x / final_time_bins, xmax=end_x / final_time_bins, color=color, linewidth=1)
        #         ax.axvline(start_x, ymin=start_y / final_freq_bins, ymax=end_y / final_freq_bins, color=color, linewidth=1)
        #         ax.axvline(end_x, ymin=start_y / final_freq_bins, ymax=end_y / final_freq_bins, color=color, linewidth=1)

        #     ax.set_xlim(0, final_time_bins)
        #     ax.set_ylim(0, final_freq_bins)
        #     ax.set_aspect(aspect='equal')

        # # Hide any unused subplots
        # for j in range(i + 1, len(axes)):
        #     axes[j].axis('off')

        # plt.tight_layout()
        # plt.show()

# generate_overlays(
#     get_data_paths = [data_root, background_path, positive_paths, negative_paths],
#     save_directory = 'spectral_detector/datasets_mutable',
#     n=3,
#     sample_rate=48000,
#     final_length_seconds=10,
#     positive_overlay_range=[7,7],
#     negative_overlay_range=[0,0],
#     save_wav=True, 
#     plot=False,
#     clear_dataset=True
#     )