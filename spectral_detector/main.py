import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
from spectrogram_tools import load_spectrogram, spectrogram_transformed, map_frequency_to_log_scale
import torchaudio, torch
from intervaltree import IntervalTree

max_time=60*60*60*2
# max_time=60*20
device = 'mps'

def interpret_files_in_directory(
        directory='../data/testing/7050014', 
        model_dir='', 
        model_nos=[25], 
        ground_truth='../data/testing/7050014/new_annotations.csv', 
        plot=False,
        to_csv=False,
        images_per_file=9, 
        max_files=1,
        confs=[0.1],
        weights_sets='best'
    ):
    all_audio_files = [file for file in os.listdir(directory) if file.endswith(('.wav', '.WAV', '.mp3', '.flac'))]
    print(f'{len(all_audio_files)} total audio files in {directory}\n')
    for model_no in model_nos:
        weights_dir = f'{model_dir}runs/detect/train{model_no}/weights/'
        # get all .pt files in the weights folder
        weights_files = [file for file in os.listdir(weights_dir) if file.endswith('.pt')]
        # TEMP remove epoch10.pt
        if weights_sets=='best':
            weights_files = [file for file in weights_files if 'best' in file]
        elif weights_sets=='epoch10':
            weights_files = [file for file in weights_files if 'epoch10' in file]
        elif weights_sets=='epoch20':
            weights_files = [file for file in weights_files if 'epoch20' in file]
        print(f'weights files: {weights_files}')
        for model_weights in weights_files:
            for conf in confs:
                timer = 0
                model = YOLO(f'{model_dir}runs/detect/train{model_no}/weights/{model_weights}')
                
                annotations = None
                if ground_truth:
                    annotations = pd.read_csv(ground_truth)

                if max_files and max_files < len(all_audio_files):
                    audio_files = all_audio_files[:max_files]
                    print(f'processing first {max_files} files')
                else: audio_files = all_audio_files
                
                per_file_metrics = {
                    'percent_centers':np.zeros(len(audio_files)),
                    'number_gt_boxes':np.zeros(len(audio_files)),
                    'percent_area_intersection':np.zeros(len(audio_files)),
                    'final_time':np.zeros(len(audio_files)),
                    'gt_time':np.zeros(len(audio_files)),
                }
                padding_values = [0, 1, 2, 3]
                for padding in padding_values:
                    per_file_metrics[f'percent_time_intersection_pad_{padding}'] = np.zeros(len(audio_files))
                    per_file_metrics[f'percent_time_fp_pad_{padding}'] = np.zeros(len(audio_files))
                
                for idx, file_name in enumerate(audio_files):
                    print(f'\n{idx}:    {file_name}')
                    if annotations is not None:
                        file_annotations = annotations['filename'] == file_name
                        this_annotations = annotations[file_annotations]
                        gt_boxes = []
                        print(f'{idx}:    {len(this_annotations)} ground truth boxes')

                    file_path = os.path.join(directory, file_name)
                    
                    spectrograms = load_spectrogram(file_path, max=images_per_file, chunk_length=10, overlap=0.5, resample_rate=48000, unit_type='power', rms_normalised=1)
                    if spectrograms==None:
                        print(f'{idx}:error Unable to load spectrograms for {file_name}. Skipping...')
                        continue
                    # send spectrograms to device

                    # convert back to waveform and save to wav for viewing testing
                    # waveform_transform = torchaudio.transforms.GriffinLim(
                    #     n_fft=2048, 
                    #     win_length=2048, 
                    #     hop_length=512, 
                    #     power=2.0
                    # )
                    # if energy_type=='dB':
                    #     normalise = 'dB_to_power'
                    # elif energy_type=='power':
                    #     normalise = None
                    # spec_audio = spectrogram_transformed(spec, normalise=normalise, to_torch=True)
                    # if energy_type=='complex':

                    # save first spectrogram as audio
                    # spec_audio = torch.square(torch.abs(spectrograms[0]))
                    # waveform = waveform_transform(spec_audio)
                    # rms = torch.sqrt(torch.mean(torch.square(waveform)))
                    # waveform = waveform*0.01/rms
                    # torchaudio.save(f"tempx.wav", waveform, sample_rate=48000)

                    # spectrograms = [spec.to(device) for spec in spectrograms]
                    timer += len(spectrograms)
                    if timer*10 > max_time:
                        print(f'Done. Processed {max_time} seconds of audio')
                        break
                    final_time = len(spectrograms)*5 + 5
                    print(f'{idx}:    converting to {len(spectrograms)} images, total time so far {timer*10}s')
                    images = []
                    for i, spec in enumerate(spectrograms):
                        spec = spectrogram_transformed(spec,
                                highpass_hz=50,
                                lowpass_hz=16000)
                        spec = spectrogram_transformed(spec,
                                set_db=-10)
                        images.append(spectrogram_transformed(spec, 
                                to_pil=True, 
                                log_scale=True, 
                                normalise='power_to_PCEN', 
                                resize=(640, 640)))
                        specific_boxes = []
                        if annotations is not None:
                            for _, row in this_annotations.iterrows():
                                # 10 seconds 50% overlap chunks
                                if (row['start_time'] >= (i*5) and row['start_time'] < ((i+2)*5)):
                                    x_start = (row['start_time'] - (i*5)) / 10
                                    y_end, y_start = map_frequency_to_log_scale(24000, [row['freq_min'], row['freq_max']])
                                    y_end = 1 - (y_end / 24000)
                                    y_start = 1 - (y_start / 24000)
                                    if row['end_time'] > ((i+2)*5):
                                        x_end = 1
                                    else:
                                        x_end = (row['end_time'] - (i*5)) / 10
                                    specific_boxes.append([x_start, y_start, x_end, y_end])
                                elif (row['end_time'] > (i*5) and row['end_time'] < ((i+2)*5)):
                                    x_start = 0
                                    y_end, y_start = map_frequency_to_log_scale(24000, [row['freq_min'], row['freq_max']])
                                    y_end = 1 - (y_end / 24000)
                                    y_start = 1 - (y_start / 24000)
                                    x_end = (row['end_time'] - (i*5)) / 10
                                    specific_boxes.append([x_start, y_start, x_end, y_end])
                            gt_boxes.append(specific_boxes)
                    if annotations is not None:
                        print(f'{idx}:    total {sum([len(box) for box in gt_boxes])} ground truth boxes in selected images')

                    results = model.predict(images, 
                        device='mps',
                        save=False, 
                        show=False,
                        verbose=False,
                        conf=conf,
                        # line_width=3, 
                        iou=0.5, # lower value for more detections
                        # visualize=False,
                    )
                    boxes = [box.xyxyn for box in (result.boxes for result in results)]
                    classes = [box.cls for box in (result.boxes for result in results)]
                    n_detections = [len(box) for box in boxes]
                    print(f'{idx}:    {sum(n_detections)} detections')


                    #TODO allow older models to be used with metrics, fix log scaling etc
                    def add_padding(interval, padding):
                        return (max(0, interval[0] - padding), interval[1] + padding)

                    def calculate_time_metrics(gt_intervals, auto_intervals, padding=0):
                        padded_auto_intervals = IntervalTree()
                        for interval in auto_intervals:
                            padded_start, padded_end = add_padding((interval.begin, interval.end), padding)
                            padded_auto_intervals.addi(padded_start, padded_end)
                        
                        padded_auto_intervals.merge_overlaps()

                        intersection_duration = 0
                        total_auto_duration = 0
                        for interval in padded_auto_intervals:
                            total_auto_duration += interval.end - interval.begin
                            overlaps = IntervalTree(gt_intervals).overlap(interval)
                            for overlap in overlaps:
                                intersection_duration += min(interval.end, overlap.end) - max(interval.begin, overlap.begin)

                        fp_duration = total_auto_duration - intersection_duration
                        return intersection_duration, fp_duration, total_auto_duration
                    
                    if (annotations is not None) and to_csv:
                        found_in_center, not_found_in_center = 0, 0
                        gt_intervals = IntervalTree()
                        auto_intervals = IntervalTree()

                        for _, row in this_annotations.iterrows():
                            start_time, end_time, start_freq, end_freq = row['start_time'], row['end_time'], row['freq_min'], row ['freq_max']
                            if (start_time > final_time) or ((end_time-start_time) < 0.1):
                                continue
                            if end_time > final_time:
                                end_time = final_time
                            start_freq, end_freq = map_frequency_to_log_scale(24000, [start_freq, end_freq])
                            gt_intervals.addi(start_time, end_time)

                            center_found = False
                            for i, box_set in enumerate(boxes):
                                for box in box_set:
                                    time_center = i*5+(((box[0] + box[2]) / 2)*10)
                                    freq_center = (1-(box[1] + box[3]) / 2) * 24000
                                    if time_center >= start_time and time_center <= end_time and freq_center >= start_freq and freq_center <= end_freq:
                                        found_in_center += 1
                                        center_found = True
                                        break
                                if center_found:
                                    break

                            if not center_found:
                                not_found_in_center += 1

                        number_gt_intervals = len(gt_intervals)
                        # Merge ground truth intervals
                        gt_intervals.merge_overlaps()
                        gt_duration = sum((interval.end - interval.begin) for interval in gt_intervals)

                        # Create automatic intervals
                        for i, box_set in enumerate(boxes):
                            for box in box_set:
                                start = i*5 + box[0]*10
                                end = i*5 + box[2]*10
                                auto_intervals.addi(start, end)

                        # Merge automatic intervals
                        auto_intervals.merge_overlaps()

                        center_percentage = 100* found_in_center / (found_in_center + not_found_in_center) if found_in_center + not_found_in_center > 0 else 0
                        per_file_metrics['percent_centers'][idx]=center_percentage
                        per_file_metrics['number_gt_boxes'][idx] = number_gt_intervals
                        per_file_metrics['final_time'][idx] = final_time
                        per_file_metrics['gt_time'][idx] = gt_duration
                        
                        # Calculate time intersection for different padding values
                        for padding in padding_values:
                            intersection_duration, fp_duration, total_auto_duration = calculate_time_metrics(gt_intervals, auto_intervals, padding)
                            
                            time_overlap_percentage = 100 * intersection_duration / gt_duration if gt_duration > 0 else 0
                            per_file_metrics[f'percent_time_intersection_pad_{padding}'][idx] = time_overlap_percentage
                            
                            fp_percentage = 100 * fp_duration / final_time if final_time > 0 else 0
                            per_file_metrics[f'percent_time_fp_pad_{padding}'][idx] = fp_percentage

                        print(f'{idx}:    {center_percentage:.2f}% of centers in {number_gt_intervals} annotations')
                        for padding in padding_values:
                            print(f'      {per_file_metrics[f"percent_time_intersection_pad_{padding}"][idx]:.2f}% time overlap with {padding}s padding')
                            print(f'      {per_file_metrics[f"percent_time_fp_pad_{padding}"][idx]:.2f}% false positive time with {padding}s padding')

                    if plot:
                        num_images = len(images)
                        num_figures = (num_images // 9) + (1 if num_images % 9 else 0)

                        for fig_num in range(num_figures):
                            fig, axs = plt.subplots(3, 3, figsize=(9, 9))
                            for i in range(9):
                                global_index = fig_num * 9 + i
                                if global_index >= num_images:
                                    break
                                image = images[global_index]
                                ax = axs[i // 3, i % 3]
                                ax.imshow(np.array(image))
                                ax.axis('off')
                                ax.title.set_text(f'seconds {global_index*5} to {(global_index+2)*5}')
                                ax.title.set_fontsize(8)
                                box_set = boxes[global_index].to('cpu')
                                species = classes[global_index].to('cpu')
                                
                                for k in range(len(box_set)):
                                    x1, y1, x2, y2 = box_set[k]
                                    x1, y1, x2, y2 = x1 * 640, y1 * 640, x2 * 640, y2 * 640

                                    if species[k] == 0:
                                        color='red'
                                    else:
                                        color='green'
                                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                                    ax.add_patch(rect)
                                
                                if annotations is not None:
                                    gt_box_set = gt_boxes[global_index]
                                    for k in range(len(gt_box_set)):
                                        x1, y1, x2, y2 = gt_box_set[k]
                                        x1, y1, x2, y2 = x1 * 640, y1 * 640, x2 * 640, y2 * 640
                                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2)
                                        ax.add_patch(rect)
                                
                                # Add horizontal lines at 100Hz intervals
                                # for freq in range(4000, 24001, 4000):
                                #     y_position = int(freq * 640 / 24000)
                                #     ax.axhline(y=y_position, color='yellow', linestyle='--', linewidth=1)

                        plt.show()

                if to_csv:
                    # Calculate global averages
                    total_gt_boxes = np.sum(per_file_metrics['number_gt_boxes'])
                    total_time = np.sum(per_file_metrics['final_time'])
                    total_gt_time = np.sum(per_file_metrics['gt_time'])

                    global_metrics = {
                        'percent_centers': np.average(per_file_metrics['percent_centers'], 
                                                    weights=per_file_metrics['number_gt_boxes'])
                    }

                    for padding in padding_values:
                        global_metrics[f'percent_time_intersection_pad_{padding}'] = np.average(
                            per_file_metrics[f'percent_time_intersection_pad_{padding}'],
                            weights=per_file_metrics['gt_time']
                        )
                        global_metrics[f'percent_time_fp_pad_{padding}'] = np.average(
                            per_file_metrics[f'percent_time_fp_pad_{padding}'],
                            weights=per_file_metrics['final_time']
                        )

                    # Prepare data for CSV
                    csv_data = {
                        'file': list(audio_files),
                        'number_gt_boxes': per_file_metrics['number_gt_boxes'].tolist(),
                        'final_time': per_file_metrics['final_time'].tolist(),
                        'gt_time': per_file_metrics['gt_time'].tolist(),
                        'percent_centers': per_file_metrics['percent_centers'].tolist()
                    }

                    for padding in padding_values:
                        csv_data[f'percent_time_intersection_pad_{padding}'] = per_file_metrics[f'percent_time_intersection_pad_{padding}'].tolist()
                        csv_data[f'percent_time_fp_pad_{padding}'] = per_file_metrics[f'percent_time_fp_pad_{padding}'].tolist()

                    # Add global metrics
                    csv_data['file'].append('GLOBAL_AVERAGE')
                    csv_data['number_gt_boxes'].append(total_gt_boxes)
                    csv_data['final_time'].append(total_time)
                    csv_data['gt_time'].append(total_gt_time)
                    csv_data['percent_centers'].append(global_metrics['percent_centers'])

                    for padding in padding_values:
                        csv_data[f'percent_time_intersection_pad_{padding}'].append(global_metrics[f'percent_time_intersection_pad_{padding}'])
                        csv_data[f'percent_time_fp_pad_{padding}'].append(global_metrics[f'percent_time_fp_pad_{padding}'])

                    # Create DataFrame and save to CSV
                    df = pd.DataFrame(csv_data)

                    # Function to format to 4 significant figures
                    def format_to_8sf(x):
                        if isinstance(x, (int, float)):
                            return f'{x:.8g}'
                        return x

                    # Apply formatting to all columns except 'file'
                    for col in df.columns:
                        if col != 'file':
                            df[col] = df[col].apply(format_to_8sf)

                    # Save to CSV
                    df.to_csv(f'results_{model_no}_{model_weights[:-3]}_conf{conf}.csv', index=False)

                    # Print global averages
                    print("\nGlobal Averages:")
                    print(f"Total Ground Truth Boxes: {total_gt_boxes}")
                    print(f"Total Time: {total_time}s")
                    print(f"Total Ground Truth Time: {total_gt_time}s")
                    print(f"Percent Centers: {global_metrics['percent_centers']:.2f}%")
                    for padding in padding_values:
                        print(f"Time Intersection (Pad {padding}s): {global_metrics[f'percent_time_intersection_pad_{padding}']:.2f}%")
                        print(f"False Positive Time (Pad {padding}s): {global_metrics[f'percent_time_fp_pad_{padding}']:.2f}%")

model_dir = 'spectral_detector/'
audio_directory = 'data/testing/7050014'
annotations_path = 'data/testing/7050014/new_annotations.csv'

# interpret_files_in_directory(audio_directory, ground_truth=annotations_path, model_dir=model_dir, model_no=25, plot=False)