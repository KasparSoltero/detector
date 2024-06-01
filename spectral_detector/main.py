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


def load_spectrograms_old(path, chunk_length=10, overlap=0.5, resample_rate=48000):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
    waveform = resample(waveform)

    # Calculate the number of samples per chunk
    samples_per_chunk = int(chunk_length * resample_rate)

    # Calculate the number of samples to overlap
    overlap_samples = int(samples_per_chunk * overlap)

    # Split the waveform into n 10-second chunks with 50% overlap
    length = waveform.shape[1]
    n = int(2*length / (samples_per_chunk)) - 1 # only works for 0.5 overlap
    if n>20: n=20
    chunks = []
    start = 0
    end = samples_per_chunk
    for _ in range(n):
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += samples_per_chunk - overlap_samples
        end += samples_per_chunk - overlap_samples

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    return [spec_transform(chunk) for chunk in chunks]

def load_spectrograms(path, chunk_length=10, overlap=0.5, resample_rate=48000):
    waveform, sample_rate = torchaudio.load(path)
    resample = torchaudio.transforms.Resample(sample_rate, resample_rate)
    waveform = resample(waveform)

    if waveform.shape[0]>1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # mono
    # rms normalise
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    waveform = waveform*1/rms

    # Calculate the number of samples per chunk
    samples_per_chunk = int(chunk_length * resample_rate)

    # Calculate the number of samples to overlap
    overlap_samples = int(samples_per_chunk * overlap)

    # Split the waveform into n 10-second chunks with 50% overlap
    length = waveform.shape[1]
    n = int(2*length / (samples_per_chunk)) - 1 # only works for 0.5 overlap
    if n>20: n=20
    chunks = []
    start = 0
    end = samples_per_chunk
    for _ in range(n):
        chunk = waveform[:, start:end]
        chunks.append(chunk)
        start += samples_per_chunk - overlap_samples
        end += samples_per_chunk - overlap_samples

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        power=2.0
    )
    return [spec_transform(chunk) for chunk in chunks]

def spec_to_image_old(spectrogram, image_normalise=0.2):
    # make ipeg image
    spec = np.squeeze(spectrogram.numpy())
    # spec = (spec - np.min(spec)) / (np.ptp(spec)) #normalise to 0-1 # the max is typically an outlier so this normalisation is destructive
    spec = spec * image_normalise / spec.mean() # normalise to 0.2 mean
    spec = np.flipud(spec) #vertical flipping for image cos

    image_width, image_height = spec.shape[1], spec.shape[0]
    image = Image.fromarray(np.uint8(spec * 255), 'L')
    image = image.resize((640, 640), Image.Resampling.LANCZOS) #resize for yolo
    return image

def spec_to_image(spectrogram):
    time_bins, freq_bins = spectrogram.shape[2], spectrogram.shape[1]
    spec = 10 * torch.log10(spectrogram + 1e-6) # to dB
    print(f'min: {torch.min(spec)}, max: {torch.max(spec)}')
    spec_clamped = spec.clamp(min=0) #clamp bottom to 0 dB
    spec_normalised = spec_clamped / spec_clamped.max() #normalise to 0-1
    spec = np.squeeze(spec_normalised.numpy())
    spec = np.flipud(spec) #vertical flipping for image cos
    value = spec # intensity of the spectrogram -> value in hsv
    saturation = 4 * value * (1 - value) # parabola for saturation
    hue = np.linspace(0,1,freq_bins)[:, np.newaxis] # linearly spaced hues over frequency range
    hue = np.tile(hue, (1,time_bins))
    
    hsv_spec = np.stack([hue, saturation, value], axis=-1)
    rgb_spec = hsv_to_rgb(hsv_spec) # convert to rgb
    # image = Image.fromarray(np.uint8(spec * 255), 'L') # greyscale
    image = Image.fromarray(np.uint8(rgb_spec * 255), 'RGB')
    image = image.resize((640, 640), Image.Resampling.LANCZOS)
    return image

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

rap_001_15_06_path = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/rapaki_all'
hinewai_path = '/Users/kaspar/Documents/ecoacoustics/AEDI/data/hinewai_2023_03_28'
lc_path = '/Users/kaspar/Documents/AEDI/data/landcarepossum_001_2159_07_02_2023_0335_11_02_2023'
lc_path_2 = '/Users/kaspar/Documents/AEDI/data/landcarepossum_001_2159_07_02_2023_0335_11_02_2023 2'
# files are named like YYYYMMDD_HHMMSS.WAV from 20230615_022500.WAV to 20230616_020700.WAV
# params
hour = 5
model_no = 19
directory = hinewai_path

# Get a list of all files in the directory
file_list = os.listdir(directory)

count=0

# Process each file individually
for file_name in file_list:
    if (file_name[-3:]=='wav') or (file_name[-3:]=='WAV'):
        # Get the hour from the file name
        file_hour = int(file_name.split('_')[1][:2])
        # Check if the file is in the specified hour
        if (file_hour == hour) or (hour == None):
            file_path = os.path.join(directory, file_name)
            print(file_name)
            count += 1
            # Process the 55 second audio file
            # Load the spectrograms
            spectrograms = load_spectrograms(file_path)
            images = [spec_to_image(spec) for spec in spectrograms]

            # model = YOLO(f'runs/detect/train{model_no}/weights/best.pt')  
            model = YOLO(f'runs/detect/train22/weights/epoch30.pt')
            results = model.predict(images, save=False, show=True)

            # process results and test
            # check_model_results(results, spectrograms, file_path)

            # plot images
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i, image in enumerate(images):
                axs[i//5, i%5].imshow(image)
                axs[i//5, i%5].axis('off')
                # title
                axs[i//5, i%5].set_title(f'{i*5} to {i*5+10} seconds')

            plt.show()

print(f'Processed {count} files')