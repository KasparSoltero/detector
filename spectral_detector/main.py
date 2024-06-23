import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
from spectrogram_tools import load_spectrogram, spectrogram_transformed, map_frequency_to_log_scale
import torchaudio, torch
from intervaltree import IntervalTree

max_time=60*60*12
# max_time=60*20
device = 'mps'

def interpret_files_in_directory(
        directory='../data/testing/7050014', 
        model_dir='', 
        model_no=25, 
        ground_truth='../data/testing/7050014/new_annotations.csv', 
        plot=False,
        images_per_file=9, 
        max_files=1
    ):
    # Get a list of all audio files in the directory
    audio_files = [file for file in os.listdir(directory) if file.endswith(('.wav', '.WAV', '.mp3', '.flac'))]
    count = 0
    model = YOLO(f'{model_dir}runs/detect/train{model_no}/weights/best.pt')
    
    annotations = None
    if ground_truth:
        annotations = pd.read_csv(ground_truth)
    print(f'{len(audio_files)} total audio files in {directory}\n')

    if max_files and max_files < len(audio_files):
        audio_files = audio_files[:max_files]
        print(f'processing first {max_files} files')
    
    per_file_metrics = {
        'percent_centers':[]*len(audio_files),
        'percent_area_intersection':[]*len(audio_files),
        'percent_time_intersection':[]*len(audio_files)
    }
    
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
        count += len(spectrograms)
        if count*10 > max_time:
            print(f'Done. Processed {max_time} seconds of audio')
            break
        final_time = len(spectrograms)*5 + 5
        print(f'{idx}:    converting to {len(spectrograms)} images, total time so far {count*10}s')
        images = []
        for i, spec in enumerate(spectrograms):
            spec = spectrogram_transformed(spec,
                    highpass_hz=50,
                    lowpass_hz=16000)
            spec = spectrogram_transformed(spec,
                    set_db=-10)
            images.append(spectrogram_transformed(spec, to_pil=True, log_scale=True, normalise='power_to_PCEN', resize=(640, 640)))
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
            conf=0.1,
            # line_width=3, 
            iou=0.5, # lower value for more detections
            # visualize=False,
        )
        boxes = [box.xyxyn for box in (result.boxes for result in results)]
        classes = [box.cls for box in (result.boxes for result in results)]
        n_detections = [len(box) for box in boxes]
        print(f'{idx}:    {sum(n_detections)} detections')

        # calculate the metrics
        def merge_intervals(intervals):
            if not intervals:
                return []
            sorted_intervals = sorted(intervals, key=lambda x: x[0])
            merged = [sorted_intervals[0]]
            for interval in sorted_intervals[1:]:
                if interval[0] <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
                else:
                    merged.append(interval)
            return merged
        
        if annotations is not None:
            found_in_center, not_found_in_center = 0, 0
            gt_intervals = []
            auto_intervals = IntervalTree()

            for _, row in this_annotations.iterrows():
                start_time, end_time, start_freq, end_freq = row['start_time'], row['end_time'], row['freq_min'], row ['freq_max']
                if start_time > final_time:
                    continue
                if end_time > final_time:
                    end_time = final_time
                start_freq, end_freq = map_frequency_to_log_scale(24000, [start_freq, end_freq])
                gt_intervals.append((start_time, end_time))

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
            gt_intervals = merge_intervals(gt_intervals)

            # Create automatic intervals
            for i, box_set in enumerate(boxes):
                for box in box_set:
                    start = i*5 + box[0]*10
                    end = i*5 + box[2]*10
                    auto_intervals.addi(start, end)

            # Merge automatic intervals
            auto_intervals.merge_overlaps()

            # Calculate intersection
            intersection_duration = 0
            for start, end in gt_intervals:
                overlaps = auto_intervals.overlap(start, end)
                for interval in overlaps:
                    intersection_duration += min(end, interval.end) - max(start, interval.begin)

            gt_duration = sum(end - start for start, end in gt_intervals)
            
            print(f'gt duration {gt_duration}, intersection duration {intersection_duration}')
            print(f'found in center: {found_in_center}, not found in center: {not_found_in_center}')
            if found_in_center + not_found_in_center != 0:
                center_percentage = 100 * found_in_center / (found_in_center + not_found_in_center)
                time_overlap_percentage = 100 * intersection_duration / gt_duration if gt_duration > 0 else 0
                print(f'{idx}:    {center_percentage:.2f}% of centers in {number_gt_intervals} annotations')
                print(f'      {time_overlap_percentage:.2f}% time overlap')
            # found_in_center, not_found_in_center = 0, 0
            # for _,row in this_annotations.iterrows():
            #     start_time, end_time, start_freq, end_freq = row['start_time'], row['end_time'], row['freq_min'], row['freq_max']
            #     start_freq, end_freq = map_frequency_to_log_scale(24000, [start_freq, end_freq])
            #     # check if there are any boxes with a center inside this box
            #     for i, box_set in enumerate(boxes):
            #         for j, box in enumerate(box_set):
            #             time_center = i*5+(((box[0] + box[2]) / 2)*10)
            #             freq_center = (1-(box[1] + box[3]) / 2) * 24000
            #             if time_center >= (start_time) and time_center <= (end_time) and freq_center >= start_freq and freq_center <= end_freq:
            #                 found_in_center+=1 # detection inside this gt box. move onto next gt box
            #                 break
            #         else:
            #             continue
            #         break
            #     else: # didn't find any detection in this gt box
            #         not_found_in_center+=1
            # if found_in_center + not_found_in_center != 0:
            #     print(f'{idx}:    {(100*found_in_center/(found_in_center+not_found_in_center)):.2f}% of centers in {len(this_annotations)} annotations')
            # else:
            #     print(f'{idx}:    no annotations...')

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

model_dir = 'spectral_detector/'
audio_directory = 'data/testing/7050014'
annotations_path = 'data/testing/7050014/new_annotations.csv'

# interpret_files_in_directory(audio_directory, ground_truth=annotations_path, model_dir=model_dir, model_no=25, plot=False)