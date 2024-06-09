import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd
import os
import torchaudio
import torch
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
# import simpleaudio as sa

from spectrogram_tools import spectrogram_transformed, load_spectrogram

# def check_model_results(results, spectrograms, file_path):
#     # display the segment of spectrogram identified
#     for i, result in enumerate(results):
#         for boxes in result.boxes:
#             for box in boxes:
#                 x1 = np.squeeze(box.xyxyn)[0]
#                 x2 = np.squeeze(box.xyxyn)[2]

#                 waveform, sample_rate = torchaudio.load(file_path)
#                 # cut waveform to x1:x2
#                 full_sample_length = waveform.shape[1]
#                 start = int(i*5*sample_rate + float(x1)*10*sample_rate)
#                 end = int(i*5*sample_rate + float(x2)*10*sample_rate)
#                 waveform = waveform[:, start:end]
#                 # play the audio
#                 waveform = waveform.numpy().transpose()
#                 audio_data = (waveform * 32767).astype("int16")  # Convert to int16 for simpleaudio compatibility

#                 # Play the audio
#                 play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)

#                 spec = np.squeeze(spectrograms[i].numpy())
#                 spec = spec * 0.2 / spec.mean() # normalise to 0.2 mean
#                 spec = np.flipud(spec) #vertical flipping for image cos
#                 # split spectrogram by x1, x2
#                 spec_width = spec.shape[1]
#                 spec_segment = spec[:, int(x1*spec_width):int(x2*spec_width)]

#                 # Stretch the spectrogram to fill the width of the figure
#                 plt.imshow(spec_segment, aspect='auto', vmin=0, vmax=1)
#                 plt.xticks(np.arange(0, spec_segment.shape[1], sample_rate), np.arange(0, spec_segment.shape[1] / sample_rate))
#                 plt.show()

# files are named like YYYYMMDD_HHMMSS.WAV from 20230615_022500.WAV to 20230616_020700.WAV
def interpret_file(filepath, hour=None, model_no=24, plot=False):

    # get directory from filepath
    directory = filepath.split('/')
    filename = directory.pop()
    directory = '/'.join(directory)
    print(directory)
    print(filename)

    # Get a list of all files in the directory
    file_list = [file for file in os.listdir(directory) if file.endswith('.wav') or file.endswith('.WAV')]

    count=0

    # if len(file_list)>50 and hour==None:
    #     file_list = random.sample(file_list, 50)
    # if filename==None:
    #     print('Sampling 50 files')
    # else:
    #     file_list = [filename]
    #     hour=None

    # # Process each file individually
    # for file_name in file_list:
    # file_path = os.path.join(directory, file_name)
    file_path=filepath
    # print(file_name)
    count += 1
    # Load the spectrograms
    spectrograms = load_spectrogram(file_path, max=20, chunk_length=10, overlap=0.5, resample_rate=48000, unit_type='complex')
    # spectrograms = [spectrogram_transformed(spec, set_snr_db=0.0001)[0] for spec in spectrograms]
    
    images=[]
    for spec in spectrograms:
        image = spectrogram_transformed(
            spec, to_pil=True, normalise='complex_to_PCEN', resize=(640, 640))
        images.append(image)
        image.show()

    model = YOLO(f'runs/detect/train{model_no}/weights/best.pt')  
    # model = YOLO(f'runs/detect/train22/weights/epoch30.pt')
    # model = YOLO(f'../runs/detect/train8/weights/best.pt')
    results = model.predict(
        images, 
        save=False, 
        show=True, 
        line_width=3, 
        conf=0.1, 
        # iou_thres=0.5,
        device='mps',
        visualize=True,
        )

    # process results and test
    # check_model_results(results, spectrograms, file_path)

    # plot images
    if plot:
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i, image in enumerate(images):
            axs[i//5, i%5].imshow(np.array(image))
            axs[i//5, i%5].axis('off')
            # title
            axs[i//5, i%5].set_title(f'{i*5} to {i*5+10} seconds')

        plt.show()

    print(f'Processed {count} files')

# process_files()